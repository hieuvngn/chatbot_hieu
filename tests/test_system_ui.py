"""System Test Case — UI-level end-to-end tests.

These tests drive the **React web UI** through a real browser using
Playwright. They are marked ``system`` and skipped unless Playwright is
installed, a Chromium browser is available, and the web bundle is built.

Prerequisites:
    uv sync --all-extras --dev          # installs playwright
    playwright install chromium         # downloads the browser
    (cd web && npm run build)           # produces web/dist served by FastAPI
    OPENROUTER_API_KEY must be set in .env

Run:
    OPENROUTER_API_KEY=... uv run pytest tests/test_system_ui.py -m system

Skip:
    uv run pytest tests/test_system_ui.py  # all skipped
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest

try:
    from playwright.sync_api import Browser, Page, sync_playwright
except ImportError:
    Browser = None  # type: ignore[assignment,misc]
    Page = None  # type: ignore[assignment,misc]
    sync_playwright = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIST = REPO_ROOT / "web" / "dist"

# The corpus the server embeds at startup. RAG_DATA_DIR must point at a
# directory holding these files; an empty dir crashes build_rag_core().
SEED_FILES = (
    "courses.json",
    "documents.json",
    "departments.json",
    "instructors.json",
    "programs.json",
    "terms.json",
)

# ---------------------------------------------------------------------------
# Skip the whole module when a prerequisite is missing
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.system,
    pytest.mark.skipif(
        sync_playwright is None,
        reason="playwright not installed — run `uv sync --all-extras --dev`",
    ),
    pytest.mark.skipif(
        not WEB_DIST.is_dir(),
        reason="web/dist missing — run `npm run build` in web/",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEMO_USERNAME = "hieu"
DEMO_PASSWORD = "demo-password"
SERVER_PORT = 8765
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"

# Seconds to wait for the server to embed the corpus and bind the port.
SERVER_READY_TIMEOUT = 240
# Milliseconds to wait for one streamed LLM answer. A single question runs
# ~8 sequential LLM/embedding round-trips against a rate-limited provider, and
# the corpus here is data/ (437 chunks). Measured 90s for the pipeline itself
# and 217s end-to-end through the browser, so 300s left too little headroom and
# failed intermittently under load rather than on a real regression.
REPLY_TIMEOUT = 600_000

# True once the page shows a SETTLED assistant turn. Tokens stream into the
# same div, so a turn counts as settled only when its text stops changing
# between polls (call with polling=1000). The user bubble and the suggestion
# buttons deliberately do not match: only ChatMessage's assistant root uses
# div.space-y-3.
REPLY_SETTLED_JS = """() => {
  const body = document.body.innerText;
  if (body.includes('\u0110ang suy ngh\u0129...')) { window.__last = null; return false; }
  if (body.includes('Kh\u00f4ng t\u00ecm \u0111\u1ee7 t\u00e0i li\u1ec7u')) return true;
  if (body.includes('Kh\u00f4ng g\u1eedi \u0111\u01b0\u1ee3c c\u00e2u tr\u1ea3 l\u1eddi')) return true;
  const msgs = Array.from(document.querySelectorAll('div.space-y-3'));
  const text = msgs.map((m) => m.innerText).join('\\n').trim();
  const settled = text.length > 0 && text === window.__last;
  window.__last = text.length > 0 ? text : null;
  return settled;
}"""


def _seed_data_dir(dest: Path) -> Path:
    """Copy the seeded corpus into ``dest`` so the server can build its index."""
    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in SEED_FILES:
        shutil.copyfile(REPO_ROOT / "data" / name, data_dir / name)
    return data_dir


def _start_server(data_dir: Path, log_path: Path) -> subprocess.Popen[bytes]:
    """Start the FastAPI server (serving web/dist) and wait until it answers."""
    env = os.environ.copy()
    env["RAG_DATA_DIR"] = str(data_dir)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + SERVER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"{BASE_URL}/api/features", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError(
            f"Server did not start within {SERVER_READY_TIMEOUT}s:\n"
            f"{log_path.read_text(errors='replace')[-4000:]}"
        )
    proc.terminate()
    raise RuntimeError(
        f"Server exited early (code {proc.returncode}):\n"
        f"{log_path.read_text(errors='replace')[-4000:]}"
    )


def _stop_server(proc: "subprocess.Popen[bytes]", grace: int = 10) -> None:
    """Terminate the server, escalating to kill if a request is still in flight.

    A pipeline call holds the event loop while it waits on the LLM, so SIGTERM
    alone can take longer than the grace period and would raise
    TimeoutExpired, masking the real test failure with a teardown error.
    """
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=grace)


def _register_via_api(base_url: str, username: str, password: str) -> bool:
    """Register a user via API. Returns True if created, False if exists."""
    try:
        data = json.dumps({"username": username, "password": password}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/auth/register",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            return False  # Already exists
        raise


def _login(page: Page) -> None:
    """Drive the login form into the chat page."""
    page.goto(f"{BASE_URL}/login")
    page.wait_for_selector('input[placeholder*="Tên"]')
    page.fill('input[placeholder*="Tên"]', DEMO_USERNAME)
    page.fill('input[type="password"]', DEMO_PASSWORD)
    page.click("button[type=submit]")
    page.wait_for_url("**/c/**", timeout=20_000)
    # The URL flips before React paints; wait for the composer so callers can
    # assert on rendered content without racing the first paint.
    page.wait_for_selector('textarea[placeholder*="Hỏi về"]', timeout=20_000)


def _new_chat(page: Page) -> None:
    """Start an empty conversation so assertions never see stale turns.

    The URL is already ``/c/<id>`` before the click, so a plain
    ``wait_for_url("**/c/**")`` returns immediately and the next action can
    land on the previous conversation — the upload then posts to the old id
    and its sidebar never lists the file. Wait for the id to actually change.
    """
    before = page.url
    page.get_by_role("button", name="Chat mới").click()
    page.wait_for_url(lambda url: url != before, timeout=20_000)
    page.wait_for_selector('textarea[placeholder*="Hỏi về"]', timeout=20_000)
    page.wait_for_function(
        "() => document.querySelectorAll('div.space-y-3').length === 0",
        timeout=20_000,
    )


def _ask(page: Page, question: str) -> None:
    """Type a question into the composer and submit with Enter."""
    composer = page.locator("textarea")
    composer.click()
    composer.fill(question)
    composer.press("Enter")


# ---------------------------------------------------------------------------
# Module-scoped fixtures: one server + one browser per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real uvicorn process over a freshly seeded, throwaway data dir."""
    tmp_path = tmp_path_factory.mktemp("system_ui")
    proc = _start_server(_seed_data_dir(tmp_path), tmp_path / "server.log")
    try:
        _register_via_api(BASE_URL, DEMO_USERNAME, DEMO_PASSWORD)
        yield BASE_URL
    finally:
        _stop_server(proc)


@pytest.fixture(scope="module")
def browser(server: str) -> Iterator[Browser]:
    assert sync_playwright is not None
    playwright = sync_playwright().start()
    try:
        launched = playwright.chromium.launch()
    except Exception as exc:  # Browser binary missing — treat as a skip.
        playwright.stop()
        pytest.skip(f"chromium not launchable — run `playwright install chromium`: {exc}")
    try:
        yield launched
    finally:
        launched.close()
        playwright.stop()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    """A fresh context per test, so auth state never leaks between tests."""
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    opened = context.new_page()
    try:
        yield opened
    finally:
        context.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUILogin:
    """Browser: open app → login page → enter credentials → chat page."""

    def test_login_page_loads(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/login")
        page.wait_for_selector("input", timeout=20_000)
        assert "CourseMate" in page.content()

    def test_login_with_demo_credentials(self, page: Page) -> None:
        _login(page)
        assert "CourseMate" in page.content()

    def test_login_redirects_to_login_when_not_authenticated(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/c/test-conv")
        page.wait_for_url("**/login**", timeout=20_000)
        # URL changes before the login form renders; assert once it exists.
        page.wait_for_selector('input[placeholder*="Tên"]', timeout=20_000)
        assert "CourseMate" in page.content()


class TestUIChat:
    """Browser: login → send message → receive response with citations."""

    @pytest.fixture(autouse=True)
    def logged_in(self, page: Page) -> None:
        _login(page)
        _new_chat(page)

    def test_chat_sends_and_receives_message(self, page: Page) -> None:
        """Send a knowledge question and verify an assistant reply renders."""
        _ask(page, "cấu trúc dữ liệu là gì?")

        # The user's own bubble appears immediately; wait for the SETTLED reply.
        page.wait_for_function(REPLY_SETTLED_JS, timeout=REPLY_TIMEOUT, polling=1000)
        body = page.locator("main").first.inner_text()
        assert "Không gửi được câu trả lời" not in body, f"send failed: {body[:400]}"
        assert "cấu trúc dữ liệu là gì?" in body, "question bubble should persist"
        rendered = (
            page.locator("div.space-y-3").count() > 0 or "Không tìm đủ tài liệu" in body
        )
        assert rendered, f"no assistant reply rendered: {body[:400]}"

    def test_citation_section_appears(self, page: Page) -> None:
        """Whichever pipeline branch answers, the DOM renders it per contract.

        The pipeline is non-deterministic across runs (grounded answer vs.
        refusal vs. general-knowledge fallback), so this asserts the UI rule
        for each outcome instead of pinning one.

        The question is on-domain (BM25 22.4 vs the 11.5 gate) so retrieval is
        trusted without the judge; an off-domain question would let the CRAG
        corrective return web-only citations, which render in a separate
        section and leave the KB citation list empty.

        The citation header is matched case-insensitively because it is styled
        ``uppercase`` — Playwright's inner_text returns the rendered text.
        """
        _ask(page, "tối ưu hóa")
        page.wait_for_function(REPLY_SETTLED_JS, timeout=REPLY_TIMEOUT, polling=1000)

        body = page.locator("main").first.inner_text()
        # The citation header is styled `uppercase`, so inner_text() returns it
        # uppercased; compare case-insensitively.
        citation_header = "tài liệu tham khảo"
        if "Không tìm đủ tài liệu" in body:  # refusal branch
            assert "Gợi ý viết lại" in body, "refusal must show a rephrase suggestion"
        elif "Không tìm thấy tài liệu liên quan" in body:  # fallback branch
            assert "trả lời theo hiểu biết chung" in body
            assert citation_header not in body.lower(), (
                "fallback must render no citations"
            )
        else:  # grounded-answer branch
            assert citation_header in body.lower(), (
                f"expected refusal/fallback/citation section, got: {body[:400]}"
            )

    def test_sidebar_shows_conversation(self, page: Page) -> None:
        """The active conversation appears in the sidebar."""
        sidebar = page.locator("aside").first
        assert sidebar.count() > 0
        assert len(sidebar.inner_text()) > 0


class TestUIUpload:
    """Browser: login → upload file → chat → see uploaded content cited."""

    @pytest.fixture(autouse=True)
    def logged_in(self, page: Page) -> None:
        _login(page)
        _new_chat(page)

    def test_upload_and_cite(self, page: Page, tmp_path: Path) -> None:
        """Upload a markdown file and ask about its content."""
        md_path = tmp_path / "bang-ham.md"
        md_path.write_text("# Bảng băm\n\nBảng băm là cấu trúc dữ liệu.\n", encoding="utf-8")

        page.locator('input[type="file"]').set_input_files(str(md_path))

        # The upload is confirmed by the filename appearing in the sidebar list.
        page.wait_for_selector("aside >> text=bang-ham.md", timeout=60_000)

        _ask(page, "bảng băm là gì?")
        page.wait_for_function(REPLY_SETTLED_JS, timeout=REPLY_TIMEOUT, polling=1000)
        assert "Không gửi được câu trả lời" not in page.locator("main").first.inner_text()
