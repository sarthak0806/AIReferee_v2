"""Polite HTTP client: per-host rate limiting, retries, size limits.

It never attempts to solve or bypass bot challenges; a challenge page is
treated as a failed request.
"""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import requests

from config import CONFIG
from utils.logging_config import get_logger

log = get_logger("http")

USER_AGENT = "AIReferee-research-collector/0.1 (academic corpus builder)"


class HttpClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        delay = CONFIG["arxiv_request_delay_s"] if "arxiv.org" in host else CONFIG["request_delay_s"]
        with self._lock:
            elapsed = time.monotonic() - self._last.get(host, 0.0)
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last[host] = time.monotonic()

    def get(self, url: str, stream: bool = False) -> requests.Response | None:
        host = urlparse(url).netloc
        for attempt in range(1, CONFIG["http_retries"] + 1):
            self._wait(host)
            try:
                resp = self.session.get(url, timeout=CONFIG["http_timeout_s"], stream=stream)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (403, 404, 410):
                    log.warning("GET %s -> %s (not retrying)", url, resp.status_code)
                    return None
                log.warning("GET %s -> %s (attempt %d)", url, resp.status_code, attempt)
            except requests.RequestException as exc:
                log.warning("GET %s failed: %s (attempt %d)", url, exc, attempt)
            time.sleep(2 ** attempt)
        return None

    def get_text(self, url: str) -> str | None:
        resp = self.get(url)
        return resp.text if resp is not None else None

    def download(self, url: str, max_bytes: int) -> bytes | None:
        resp = self.get(url, stream=True)
        if resp is None:
            return None
        buf = bytearray()
        try:
            for part in resp.iter_content(65536):
                buf.extend(part)
                if len(buf) > max_bytes:
                    log.warning("Download %s exceeded %d bytes; aborted", url, max_bytes)
                    return None
        except requests.RequestException as exc:
            log.warning("Download %s interrupted: %s", url, exc)
            return None
        return bytes(buf)


_client: HttpClient | None = None


def client() -> HttpClient:
    global _client
    if _client is None:
        _client = HttpClient()
    return _client
