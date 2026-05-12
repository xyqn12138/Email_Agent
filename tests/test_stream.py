import sys, asyncio
sys.path.insert(0, 'src')
from agent.graph import build_graph

async def test():
    g = build_graph()
    # 用一个会触发 tool 调用的问题
    inp = {'messages': [{'role': 'user', 'content': '什么是进程同步'}]}
    count = 0
    async for event in g.astream_events(inp, version='v2'):
        kind = event.get('event', '')
        count += 1
        if count > 80:
            print('... truncated')
            break
        extra = ''
        if kind == 'on_tool_start':
            extra = f' -> {event.get("name", "")}'
        elif kind == 'on_chat_model_stream':
            chunk = event.get('chunk')
            if chunk and hasattr(chunk, 'content'):
                extra = f' -> {chunk.content!r:.50}'
        elif kind == 'on_chat_model_end':
            extra = ' -> LLM done'
        elif kind == 'on_tool_end':
            output = event.get('data', {}).get('output', '')
            extra = f' -> tool done ({len(str(output))} chars)'
        print(f'{count:3d} {kind}{extra}')
    print(f'Total events: {count}')

asyncio.run(test())
