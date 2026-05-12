import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.graph import build_graph
from agent.rag.data_embedding import RAGPipelineService
from agent.utils.logger_handler import get_logger
from agent.utils.path_handler import get_absolute_path

load_dotenv()
logger = get_logger()

# --- Paths ---
DATA_DIR = Path(get_absolute_path("data"))
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_FILE = Path(get_absolute_path("knowledge.md"))
STATIC_DIR = Path(get_absolute_path("static"))

# --- App ---
app = FastAPI(title="Study Agent")

# --- Graph (lazy init) ---
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ============================================================
# Knowledge.md
# ============================================================
def _add_knowledge_entry(filename: str, chunks: int):
    if not KNOWLEDGE_FILE.exists():
        KNOWLEDGE_FILE.write_text(
            "# 知识库课本列表\n\n"
            "| 序号 | 文件名 | 片段数 | 上传时间 |\n"
            "|------|--------|--------|----------|\n",
            encoding="utf-8",
        )
    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    existing = [l for l in content.splitlines() if l.startswith("|") and "序号" not in l and "---" not in l]
    idx = len(existing) + 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = f"| {idx} | {filename} | {chunks} | {now} |\n"
    if not content.endswith("\n"):
        content += "\n"
    content += new_row
    KNOWLEDGE_FILE.write_text(content, encoding="utf-8")


# ============================================================
# API Routes
# ============================================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    dest = DATA_DIR / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loop = asyncio.get_event_loop()
    try:
        service = RAGPipelineService(model_name="dashscope")
        try:
            prepared = await loop.run_in_executor(None, service.ingest_file, str(dest))
            chunks_count = len(prepared.chunks)
        finally:
            service.close()

        _add_knowledge_entry(filename, chunks_count)
        return {"status": "ok", "filename": filename, "chunks": chunks_count}
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/knowledge")
async def get_knowledge():
    if not KNOWLEDGE_FILE.exists():
        return {"content": "暂无已入库的课本"}
    return {"content": KNOWLEDGE_FILE.read_text(encoding="utf-8")}


TOOL_NAME_MAP = {
    "knowledge_base_search": "检索知识库",
    "fetch_neighbor_context": "获取上下文片段",
    "view_image": "查看图片",
    "web_search": "搜索互联网",
}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "empty message"})

    graph = _get_graph()
    input_msg = {"messages": [{"role": "user", "content": message}]}
    config = {"recursion_limit": 50}

    async def event_stream():
        got_tokens = False
        final_content = ""

        try:
            async for event in graph.astream_events(input_msg, config=config, version="v2"):
                kind = event.get("event", "")

                if kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except Exception:
                            pass
                    display_name = TOOL_NAME_MAP.get(tool_name, tool_name)
                    detail = ""
                    if isinstance(tool_input, dict):
                        for v in tool_input.values():
                            if isinstance(v, str) and len(v) > 2:
                                detail = v[:120]
                                break
                    elif isinstance(tool_input, str):
                        detail = tool_input[:120]
                    yield _sse("thinking", {"tool": display_name, "detail": detail})

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    if hasattr(output, "content"):
                        output = output.content or ""
                    if not isinstance(output, str):
                        output = str(output)[:200]
                    yield _sse("tool_result", {"output": output[:200]})

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    text = ""
                    if isinstance(chunk, dict):
                        text = chunk.get("content", "")
                    elif hasattr(chunk, "content"):
                        text = chunk.content or ""
                    if text:
                        got_tokens = True
                        yield _sse("token", {"text": text})

                elif kind == "on_chat_model_end":
                    # Capture final content from LLM output as fallback
                    output = event.get("data", {}).get("output", {})
                    if hasattr(output, "content"):
                        final_content = output.content or ""
                    elif isinstance(output, dict):
                        final_content = output.get("content", "")

                elif kind == "on_chain_end":
                    # Capture from chain output if still no tokens
                    if not got_tokens and not final_content:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            msgs = output.get("messages", [])
                            if msgs:
                                last = msgs[-1]
                                if hasattr(last, "content"):
                                    final_content = last.content or ""
                                elif isinstance(last, dict):
                                    final_content = last.get("content", "")

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield _sse("error", {"message": str(e)})

        # Fallback: if no tokens were streamed but we have final content
        if not got_tokens and final_content:
            logger.warning("No streaming tokens received, using fallback final content")
            # Send in chunks to simulate streaming
            chunk_size = 10
            for i in range(0, len(final_content), chunk_size):
                yield _sse("token", {"text": final_content[i:i + chunk_size]})

        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --- Serve frontend ---
@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8080, reload=True)
