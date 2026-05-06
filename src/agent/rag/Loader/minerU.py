import io
import os
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from agent.utils.logger_handler import get_logger
from agent.utils.path_handler import get_absolute_path

load_dotenv()

logger = get_logger()

DEFAULT_BASE_URL = "https://mineru.net"
POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 360


class MinerUParser:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_version: str = "vlm",
        is_ocr: bool = False,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "ch",
    ):
        self.api_key = api_key or os.getenv("MINERU_API_KEY")
        if not self.api_key:
            raise ValueError("MINERU_API_KEY is required. Set it in .env or pass api_key parameter.")
        raw_url = base_url or os.getenv("MINERU_BASE_URL", DEFAULT_BASE_URL)
        self.base_url = self._extract_base_domain(raw_url)
        self.model_version = model_version
        self.is_ocr = is_ocr
        self.enable_formula = enable_formula
        self.enable_table = enable_table
        self.language = language
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _extract_base_domain(raw_url: str) -> str:
        url = raw_url.rstrip("/")
        for path_segment in ("/api/v4", "/api/v1"):
            idx = url.find(path_segment)
            if idx > 0:
                url = url[:idx]
                break
        return url.rstrip("/")

    def _request_upload_url(self, filename: str) -> tuple[str, str]:
        url = f"{self.base_url}/api/v4/file-urls/batch"
        payload = {
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
            "model_version": self.model_version,
            "files": [
                {
                    "name": filename,
                    "is_ocr": self.is_ocr,
                }
            ],
        }
        resp = requests.post(url, headers=self._headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logger.error(f"MinerU API returned {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU upload URL request failed: {result.get('msg')}")
        batch_id: str = result["data"]["batch_id"]
        upload_url: str = result["data"]["file_urls"][0]
        return batch_id, upload_url

    def _upload_file(self, upload_url: str, file_path: Path) -> None:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, data=f, timeout=120)
        resp.raise_for_status()

    def _poll_batch_result(self, batch_id: str) -> dict:
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
            resp = requests.get(url, headers=self._headers, timeout=30)
            if resp.status_code != 200:
                logger.error(f"MinerU poll returned {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") != 0:
                raise RuntimeError(f"MinerU poll failed: {result.get('msg')}")

            extract_results = result["data"].get("extract_result", [])
            if not extract_results:
                logger.info(f"MinerU task queued, waiting... attempt {attempt}/{MAX_POLL_ATTEMPTS}")
                time.sleep(POLL_INTERVAL)
                continue

            task = extract_results[0]
            state = task.get("state", "")

            if state == "done":
                return task
            elif state == "failed":
                raise RuntimeError(f"MinerU extraction failed: {task.get('err_msg')}")
            else:
                progress = task.get("extract_progress", {})
                if progress:
                    extracted = progress.get("extracted_pages", 0)
                    total = progress.get("total_pages", 0)
                    logger.info(
                        f"MinerU progress: {extracted}/{total} pages, state={state}"
                    )
                else:
                    logger.info(f"MinerU state={state}, waiting... attempt {attempt}/{MAX_POLL_ATTEMPTS}")
                time.sleep(POLL_INTERVAL)

        raise TimeoutError("MinerU extraction timed out")

    def _download_and_extract_zip(self, zip_url: str, output_dir: Path) -> None:
        resp = requests.get(zip_url, timeout=120)
        resp.raise_for_status()
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(output_dir)

    @staticmethod
    def _find_md_file(directory: Path) -> Path | None:
        for md_file in directory.rglob("*.md"):
            return md_file
        return None

    def parse(self, file_path: str, output_dir: str | None = None) -> Path:
        absolute_path = Path(get_absolute_path(file_path))
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {absolute_path}")

        filename = absolute_path.name
        stem = absolute_path.stem

        if output_dir is None:
            target_dir = Path(get_absolute_path(f"data/{stem}"))
        else:
            target_dir = Path(output_dir)

        logger.info(f"Uploading '{filename}' to MinerU...")
        batch_id, upload_url = self._request_upload_url(filename)
        self._upload_file(upload_url, absolute_path)
        logger.info(f"File uploaded successfully, batch_id={batch_id}")

        logger.info("Waiting for MinerU extraction to complete...")
        task_result = self._poll_batch_result(batch_id)

        zip_url = task_result.get("full_zip_url")
        if not zip_url:
            raise RuntimeError("MinerU task completed but no download URL was returned")

        logger.info(f"Downloading extraction results to '{target_dir}'...")
        self._download_and_extract_zip(zip_url, target_dir)

        md_file = self._find_md_file(target_dir)
        if md_file:
            logger.info(f"Markdown file saved: {md_file}")
        else:
            logger.warning("No markdown file found in the extraction archive")

        return target_dir


if __name__ == "__main__":
    parser = MinerUParser()
    result_dir = parser.parse(r"src\agent\data\test.pdf")
    print(f"Results saved to: {result_dir}")