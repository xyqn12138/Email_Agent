import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import pypdf
import requests
from dotenv import load_dotenv

from agent.utils.logger_handler import get_logger
from agent.utils.path_handler import get_absolute_path

load_dotenv()

logger = get_logger()

DEFAULT_BASE_URL = "https://mineru.net"
POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 360
MAX_PAGES_PER_REQUEST = 100


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

    def _request_upload_url(self, filename: str, page_ranges: str | None = None) -> tuple[str, str]:
        url = f"{self.base_url}/api/v4/file-urls/batch"
        file_entry: dict = {
            "name": filename,
            "is_ocr": self.is_ocr,
        }
        if page_ranges:
            file_entry["page_ranges"] = page_ranges
        payload = {
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
            "model_version": self.model_version,
            "files": [file_entry],
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
        size_mb = file_path.stat().st_size / (1024 * 1024)
        timeout = max(300, int(size_mb * 5))  # 5s per MB, minimum 300s
        logger.info(f"Uploading {file_path.name} ({size_mb:.1f}MB), timeout={timeout}s")
        for attempt in range(3):
            try:
                with open(file_path, "rb") as f:
                    resp = requests.put(upload_url, data=f, timeout=timeout)
                resp.raise_for_status()
                return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < 2:
                    wait = (attempt + 1) * 10
                    logger.warning(f"Upload failed (attempt {attempt+1}/3), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

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
    def _get_pdf_page_count(file_path: Path) -> int:
        reader = pypdf.PdfReader(str(file_path))
        return len(reader.pages)

    @staticmethod
    def _build_page_ranges(total_pages: int, chunk_size: int = MAX_PAGES_PER_REQUEST) -> list[str]:
        ranges = []
        for start in range(1, total_pages + 1, chunk_size):
            end = min(start + chunk_size - 1, total_pages)
            ranges.append(f"{start}-{end}")
        return ranges

    @staticmethod
    def _find_md_file(directory: Path) -> Path | None:
        for md_file in directory.rglob("*.md"):
            return md_file
        return None

    def _parse_single(self, file_path: Path, output_dir: Path, page_ranges: str | None = None) -> Path:
        filename = file_path.name
        logger.info(f"Uploading '{filename}' to MinerU" + (f" (pages: {page_ranges})" if page_ranges else "") + "...")
        batch_id, upload_url = self._request_upload_url(filename, page_ranges)
        self._upload_file(upload_url, file_path)
        logger.info(f"File uploaded successfully, batch_id={batch_id}")

        logger.info("Waiting for MinerU extraction to complete...")
        task_result = self._poll_batch_result(batch_id)

        zip_url = task_result.get("full_zip_url")
        if not zip_url:
            raise RuntimeError("MinerU task completed but no download URL was returned")

        logger.info(f"Downloading extraction results to '{output_dir}'...")
        self._download_and_extract_zip(zip_url, output_dir)

        md_file = self._find_md_file(output_dir)
        if md_file:
            logger.info(f"Markdown file saved: {md_file}")
        else:
            logger.warning("No markdown file found in the extraction archive")

        return output_dir

    @staticmethod
    def _merge_markdowns(part_dirs: list[Path], merged_md_path: Path) -> None:
        contents = []
        for part_dir in part_dirs:
            md_file = None
            for candidate in part_dir.rglob("*.md"):
                md_file = candidate
                break
            if md_file:
                text = md_file.read_text(encoding="utf-8")
                contents.append(text)
            else:
                logger.warning(f"No markdown file found in {part_dir}")

        merged_md_path.parent.mkdir(parents=True, exist_ok=True)
        merged_md_path.write_text("\n\n".join(contents), encoding="utf-8")
        logger.info(f"Merged markdown saved: {merged_md_path}")

    def parse(self, file_path: str, output_dir: str | None = None) -> Path:
        absolute_path = Path(get_absolute_path(file_path))
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {absolute_path}")

        stem = absolute_path.stem

        if output_dir is None:
            target_dir = Path(get_absolute_path(f"data/{stem}"))
        else:
            target_dir = Path(output_dir)

        if absolute_path.suffix.lower() == ".pdf":
            total_pages = self._get_pdf_page_count(absolute_path)
            logger.info(f"PDF has {total_pages} pages")

            if total_pages > MAX_PAGES_PER_REQUEST:
                return self._parse_large_pdf(absolute_path, target_dir, total_pages)

        self._parse_single(absolute_path, target_dir)
        return target_dir

    @staticmethod
    def _split_pdf(file_path: Path, page_ranges: list[str], temp_dir: Path) -> list[Path]:
        """Split a PDF into physical files by page ranges."""
        reader = pypdf.PdfReader(str(file_path))
        part_files = []
        for idx, pr in enumerate(page_ranges, 1):
            start, end = pr.split("-")
            start_page = int(start) - 1  # 0-indexed
            end_page = int(end)          # inclusive in pypdf
            writer = pypdf.PdfWriter()
            for p in range(start_page, end_page):
                writer.add_page(reader.pages[p])
            part_path = temp_dir / f"{file_path.stem}_part{idx}.pdf"
            with open(part_path, "wb") as f:
                writer.write(f)
            part_files.append(part_path)
            size_mb = part_path.stat().st_size / (1024 * 1024)
            logger.info(f"Split part {idx}: pages {pr}, {size_mb:.1f}MB -> {part_path.name}")
        return part_files

    def _parse_large_pdf(self, file_path: Path, target_dir: Path, total_pages: int) -> Path:
        page_ranges = self._build_page_ranges(total_pages, MAX_PAGES_PER_REQUEST)
        num_parts = len(page_ranges)
        logger.info(f"PDF exceeds {MAX_PAGES_PER_REQUEST} pages, splitting into {num_parts} batches")

        temp_dir = Path(tempfile.mkdtemp(prefix="mineru_split_"))
        part_dirs: list[Path] = []

        try:
            # Physically split PDF to avoid MinerU file size limit
            part_files = self._split_pdf(file_path, page_ranges, temp_dir)

            for idx, (part_file, pr) in enumerate(zip(part_files, page_ranges), 1):
                part_output = target_dir / f"part_{idx}"
                part_dirs.append(part_output)
                logger.info(f"--- Processing batch {idx}/{num_parts}: pages {pr} ---")
                self._parse_single(part_file, part_output)

            merged_md = target_dir / f"{file_path.stem}.md"
            self._merge_markdowns(part_dirs, merged_md)

            self._consolidate_images(part_dirs, target_dir)

            self._merge_content_lists(part_dirs, page_ranges, target_dir)

        finally:
            for part_dir in part_dirs:
                if part_dir.exists():
                    shutil.rmtree(part_dir, ignore_errors=True)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info(f"All batches merged into: {target_dir}")
        return target_dir

    @staticmethod
    def _consolidate_images(part_dirs: list[Path], target_dir: Path) -> None:
        images_dir = target_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for part_dir in part_dirs:
            part_images = part_dir / "images"
            if not part_images.exists():
                continue
            for img in part_images.iterdir():
                if img.is_file():
                    dest = images_dir / img.name
                    if not dest.exists():
                        shutil.copy2(str(img), str(dest))

    @staticmethod
    def _find_content_list(directory: Path) -> Path | None:
        direct = directory / "content_list.json"
        if direct.exists():
            return direct
        for match in directory.glob("*_content_list.json"):
            return match
        return None

    @staticmethod
    def _merge_content_lists(
        part_dirs: list[Path],
        page_ranges: list[str],
        target_dir: Path,
    ) -> None:
        merged_items: list[dict] = []
        for part_dir, page_range in zip(part_dirs, page_ranges):
            cl_path = MinerUParser._find_content_list(part_dir)
            if not cl_path:
                logger.warning(f"No content_list.json found in {part_dir}")
                continue
            with open(cl_path, encoding="utf-8") as f:
                items = json.load(f)
            start_page = int(page_range.split("-")[0]) - 1
            for item in items:
                item["page_idx"] = item.get("page_idx", 0) + start_page
            merged_items.extend(items)

        if not merged_items:
            logger.warning("No content_list items collected from any batch")
            return

        output_path = target_dir / "content_list.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Merged content_list saved: {output_path} ({len(merged_items)} items)")


if __name__ == "__main__":
    parser = MinerUParser()
    result_dir = parser.parse(r"src\agent\data\计算机操作系统  第4版·微课视频版_9787302577614_15189771.pdf")
    print(f"Results saved to: {result_dir}")
