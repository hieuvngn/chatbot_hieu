"""Contract tests binding the React client to the FastAPI backend.

The web app has no unit tests, so nothing else guards the two sides from
drifting apart: a renamed response field, a moved endpoint, an unhandled
stream event, or a request body key the backend silently ignores. This
suite parses ``web/src/lib/types.ts`` and ``web/src/lib/api.ts`` and
asserts they still describe what ``server/schemas.py`` and
``server/routes.py`` accept and return.

Static checks cover the whole surface. The streaming checks drive the real
endpoint through ``TestClient`` because the NDJSON bodies are assembled by
hand in ``routes.chat_stream`` — the easiest place for a field to go
missing, and invisible to a plain source comparison.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_core import StreamDelta, StreamDone, StreamRefused, StreamStart
from rag_core.attachments import MAX_FILES_PER_CONVERSATION
from rag_core.models import (
    AnswerResult,
    Citation,
    Department,
    EntityBundle,
    Instructor,
    Program,
    Session,
    Source,
    Term,
)
from server.main import create_app
from server.schemas import (
    AttachmentOut,
    ChatIn,
    ConversationCreateIn,
    ConversationOut,
    ConversationRenameIn,
    CitationOut,
    EntitiesOut,
    EntityOut,
    FeaturesOut,
    ProfileIn,
    RegisterIn,
    SourceOut,
    TokenOut,
    TurnOut,
    UserOut,
)
from server.state import AppState
from tests.test_server_api import make_client

ROOT = Path(__file__).resolve().parent.parent
TYPES_TS = ROOT / "web" / "src" / "lib" / "types.ts"
API_TS = ROOT / "web" / "src" / "lib" / "api.ts"
ROUTES_PY = ROOT / "server" / "routes.py"
ATTACHMENTS_TSX = ROOT / "web" / "src" / "components" / "AttachmentsSection.tsx"

needs_web = pytest.mark.skipif(
    not TYPES_TS.exists(), reason="web/src not present — no client to contract"
)

# Response models the UI renders, keyed by the TS interface that backs them.
RESPONSE_MODELS: dict[str, type[Any]] = {
    "User": UserOut,
    "Source": SourceOut,
    "Citation": CitationOut,
    "Turn": TurnOut,
    "ConversationMeta": ConversationOut,
    "Attachment": AttachmentOut,
    "Features": FeaturesOut,
    "Entity": EntityOut,
    "EntityBundle": EntitiesOut,
    "AuthResponse": TokenOut,
}

# Request bodies the UI sends, keyed by (method, normalized path in api.ts).
REQUEST_MODELS: dict[tuple[str, str], type[Any]] = {
    ("POST", "/api/auth/register"): RegisterIn,
    ("POST", "/api/auth/login"): RegisterIn,
    ("PATCH", "/api/users/me"): ProfileIn,
    ("POST", "/api/conversations"): ConversationCreateIn,
    ("PATCH", "/api/conversations/{}"): ConversationRenameIn,
    ("POST", "/api/conversations/{}/chat"): ChatIn,
    ("POST", "/api/conversations/{}/chat/stream"): ChatIn,
}

_INTERFACE_RE = re.compile(r"export interface (\w+)\s*\{(.*?)\n\}", re.DOTALL)
_FIELD_RE = re.compile(r"^\s*(\w+)(\?)?\s*[:(]", re.MULTILINE)


def parse_interfaces(source: str) -> dict[str, dict[str, bool]]:
    """Map ``interface Name { field: T }`` to {field: is_required}.

    A ``?`` marks the field optional. Signature members (parenthesised
    params) are not fields, hence the ``[:(]`` tail in the field pattern.
    """
    return {
        match.group(1): {
            field.group(1): field.group(2) is None
            for field in _FIELD_RE.finditer(match.group(2))
        }
        for match in _INTERFACE_RE.finditer(source)
    }


def required_fields(source: str, interface: str) -> set[str]:
    """Field names the interface declares as required (no ``?``)."""
    fields = parse_interfaces(source).get(interface)
    assert fields, f"{interface} not found — the parser or the client types changed shape"
    return {name for name, is_required in fields.items() if is_required}


def client_interfaces() -> str:
    """Both client files, so a name resolves against whichever declares it."""
    return TYPES_TS.read_text(encoding="utf-8") + "\n" + API_TS.read_text(encoding="utf-8")


def ts_interfaces() -> str:
    return TYPES_TS.read_text(encoding="utf-8")


def _call_arguments(source: str, callee: str) -> list[str]:
    """Return the argument text of every ``callee(`` call taking a literal path."""
    calls: list[str] = []
    for match in re.finditer(callee + r"(?:<[^>]*>)?\(", source):
        # Skip declarations: the first non-space argument must be a string literal.
        if source[match.end() :].lstrip()[:1] not in {'"', "`"}:
            continue
        start = match.end() - 1
        depth = 0
        end = len(source)
        for i in range(start, len(source)):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        else:
            raise AssertionError(f"unbalanced {callee}( call at offset {match.start()}")
        calls.append(source[start + 1 : end])
    return calls


def normalize_path(path: str) -> str:
    """Collapse params and query strings into one comparable template.

    ``/conversations/${id}/chat?x=1`` (api.ts) and
    ``/conversations/{conversation_id}/chat`` (FastAPI) both become
    ``/conversations/{}/chat``.
    """
    path = path.split("?")[0]
    path = re.sub(r"\$\{[^}]*\}", "{}", path)
    return re.sub(r"\{[^}]*\}", "{}", path)


def parse_api_calls() -> list[tuple[str, str, set[str]]]:
    """Map each api.ts HTTP call to (method, path, json body keys).

    ``request()`` takes a path relative to ``/api``; ``streamChat`` calls
    ``fetch()`` with the prefix already spelled out, so each keeps its own.
    """
    source = API_TS.read_text(encoding="utf-8")
    calls: list[tuple[str, str, set[str]]] = []
    for callee, prefix in (("request", "/api"), ("fetch", "")):
        for arguments in _call_arguments(source, callee):
            path_match = re.match(r"\s*[`\"]([^`\"]+)[`\"]", arguments)
            assert path_match is not None, f"cannot read path from {arguments!r}"
            literal = path_match.group(1)
            # request()'s own `fetch(`/api${path}`)` passthrough names no endpoint.
            if re.sub(r"\$\{[^}]*\}", "", literal).rstrip("/") in {"", "/api"}:
                continue
            method = re.search(r"method:\s*\"(\w+)\"", arguments)
            body = re.search(r"JSON\.stringify\(\{(.*?)\}\)", arguments, re.DOTALL)
            keys = set(re.findall(r"(\w+)\s*[:,}]", body.group(1))) if body else set()
            calls.append(
                (
                    method.group(1) if method else "GET",
                    normalize_path(prefix + path_match.group(1)),
                    keys,
                )
            )
    return calls


def server_routes() -> set[tuple[str, str]]:
    """(method, path) pairs the mounted API serves, read off the OpenAPI schema.

    ``app.routes`` wraps included routers in a private container, so the
    schema is the stable public view of the same surface.
    """
    paths = create_app().openapi()["paths"]
    return {
        (method.upper(), normalize_path(path))
        for path, operations in paths.items()
        for method in operations
        if method not in {"head", "options"}
    }


@needs_web
@pytest.mark.parametrize("interface", sorted(RESPONSE_MODELS))
def test_ui_required_fields_exist_in_backend_models(interface: str) -> None:
    """Every field the UI reads as required must exist on the Pydantic model.

    This is the rename guard: dropping ``rephrase_suggestion`` from
    ``TurnOut`` while ``types.ts`` still reads it leaves the UI rendering
    ``undefined``, and no other suite would notice.
    """
    required = required_fields(client_interfaces(), interface)
    missing = required - set(RESPONSE_MODELS[interface].model_fields)
    assert not missing, f"{interface}: UI requires {sorted(missing)}, the model omits them"


@needs_web
def test_every_api_call_reaches_a_server_route() -> None:
    """Each api.ts call must resolve to a real (method, path) on the API."""
    routes = server_routes()
    unknown = [
        (method, path) for method, path, _ in parse_api_calls() if (method, path) not in routes
    ]
    assert not unknown, f"api.ts calls endpoints the server does not serve: {unknown}"


@needs_web
@pytest.mark.parametrize(
    "model_key", sorted(REQUEST_MODELS), ids=lambda key: f"{key[0]} {key[1]}"
)
def test_request_bodies_only_send_accepted_fields(model_key: tuple[str, str]) -> None:
    """Body keys api.ts sends must be fields the endpoint's model accepts.

    A typo'd key is dropped by Pydantic, so the UI would send
    ``useWeb: false`` and silently get the server default instead.
    """
    method, path = model_key
    accepted = set(REQUEST_MODELS[model_key].model_fields)
    sent = [keys for m, p, keys in parse_api_calls() if (m, p) == (method, path)]
    assert sent, f"no api.ts call for {method} {path} — REQUEST_MODELS is stale"
    for keys in sent:
        assert not keys - accepted, f"{method} {path} sends rejected {sorted(keys - accepted)}"


@needs_web
def test_upload_form_fields_match_endpoint_signature() -> None:
    """FormData keys must match the upload endpoint's parameter names."""
    sent = set(re.findall(r'fd\.append\("(\w+)"', API_TS.read_text(encoding="utf-8")))
    source = ROUTES_PY.read_text(encoding="utf-8")
    upload = source[source.index("async def upload_attachment") :]
    expected = set(
        re.findall(r"(\w+):\s*(?:Annotated\[\w+,\s*Form\(\)\]|UploadFile)", upload)
    )
    assert sent == expected, f"FormData keys {sorted(sent)} != endpoint {sorted(expected)}"


@needs_web
def test_stream_event_names_are_all_handled() -> None:
    """Every NDJSON event the server emits must be handled by streamChat.

    An unhandled event is silently ignored, so the UI would keep showing
    "Đang suy nghĩ…" forever waiting for a ``done`` its switch never sees.
    """
    handled = set(re.findall(r'case "(\w+)":', API_TS.read_text(encoding="utf-8")))
    emitted = set(re.findall(r'"event":\s*"(\w+)"', ROUTES_PY.read_text(encoding="utf-8")))
    assert emitted, "no event names found in routes.py — emitter moved?"
    assert emitted <= handled, f"server emits unhandled events: {sorted(emitted - handled)}"


@needs_web
def test_attachment_limit_matches_backend_limit() -> None:
    """AttachmentsSection hard-codes MAX_FILES; the backend owns the number.

    If the UI hides the upload button below the backend's cap, users can
    never reach the limit; above it, uploads fail with a 400.
    """
    declared = re.search(r"const MAX_FILES = (\d+)", ATTACHMENTS_TSX.read_text(encoding="utf-8"))
    assert declared is not None, "MAX_FILES constant missing from AttachmentsSection"
    assert int(declared.group(1)) == MAX_FILES_PER_CONVERSATION


# ---------------------------------------------------------------------------
# Runtime: routes.chat_stream assembles its NDJSON bodies by hand
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Unit vectors; the attachment store in AppState only needs the shape."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class StreamingCore:
    """CoreLike stub emitting one known answer — no LLM, no network."""

    has_web_search = False
    embedder = StubEmbedder()

    def __init__(self) -> None:
        self.source = Source(
            document_id="DOC-001",
            document_title="Slides Cấu trúc dữ liệu",
            chapter="Chương 3: Bảng băm",
            course_code="CS101",
            kind="slides",
            language="vi",
        )

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult:
        return self._result()

    def stream_answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> Any:
        yield StreamStart(skills_applied=["eli5"], sources=[self.source])
        yield StreamDelta(delta="Bảng băm là ")
        yield StreamDelta(delta="cấu trúc dữ liệu tra cứu khóa–giá trị.")
        yield StreamDone(result=self._result())

    def _result(self) -> AnswerResult:
        return AnswerResult(
            answer="Bảng băm là cấu trúc dữ liệu tra cứu khóa–giá trị.",
            citations=[Citation(marker="[1]", source=self.source)],
            sources=[self.source],
        )

    @property
    def entities(self) -> EntityBundle:
        """One record per collection, so /entities has a known shape."""
        return EntityBundle(
            departments=(Department(id="DEP-CNTT", name="Khoa CNTT", name_en="CS"),),
            instructors=(
                Instructor(
                    id="INS-1",
                    name="TS. Nguyễn Văn A",
                    title="Tiến sĩ",
                    email="a@sv.edu.vn",
                    department_id="DEP-CNTT",
                    bio="",
                    courses=(("CS101", "giảng viên"),),
                ),
            ),
            programs=(
                Program(
                    id="PR-CNTT",
                    name="Kỹ sư CNTT",
                    name_en="CS Engineer",
                    department_id="DEP-CNTT",
                    total_credits=140,
                ),
            ),
            terms=(
                Term(
                    id="T-2025-S1",
                    name="Học kỳ 1 năm 2025",
                    year=2025,
                    season="1",
                    start_date="2025-09-01",
                    end_date="2025-12-31",
                ),
            ),
        )

def streaming_client(tmp_path: Path) -> tuple[TestClient, dict[str, dict[str, str]]]:
    client, _, _, headers = make_client(tmp_path)
    state = cast(AppState, cast(FastAPI, client.app).state.coursemate)
    state.core = StreamingCore()
    return client, headers


def stream_events(
    client: TestClient, headers: dict[str, str], conv_id: str
) -> list[dict[str, Any]]:
    response = client.post(
        f"/api/conversations/{conv_id}/chat/stream",
        json={"message": "bảng băm là gì?"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.strip().split("\n") if line]


@needs_web
def test_done_event_carries_every_field_the_client_reads(tmp_path: Path) -> None:
    """The done event must satisfy the required fields of StreamDone."""
    client, headers = streaming_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=headers["alice"]).json()
    done = next(e for e in stream_events(client, headers["alice"], conv["id"]) if e["event"] == "done")

    declared = parse_interfaces(API_TS.read_text(encoding="utf-8"))["StreamDone"]
    missing = {name for name, is_required in declared.items() if is_required} - set(done)
    assert not missing, f"done event lacks {sorted(missing)}: has {sorted(done)}"


@needs_web
def test_start_event_sources_match_stream_source(tmp_path: Path) -> None:
    """Sources in the start event must satisfy the required StreamSource fields."""
    client, headers = streaming_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=headers["alice"]).json()
    start = next(e for e in stream_events(client, headers["alice"], conv["id"]) if e["event"] == "start")

    assert start["skills_applied"] == ["eli5"], "start must report applied skills"
    assert start["sources"], "start must carry sources for the UI to preview"
    required = required_fields(API_TS.read_text(encoding="utf-8"), "StreamSource")
    missing = required - set(start["sources"][0])
    assert not missing, f"stream source lacks {sorted(missing)}: has {sorted(start['sources'][0])}"


@needs_web
def test_streamed_tokens_reassemble_into_the_final_answer(tmp_path: Path) -> None:
    """Deltas must concatenate to exactly the done answer.

    The UI renders deltas live, then replaces the bubble with the done
    answer after refetching; a mismatch makes the text visibly change
    between the streaming and settled states.
    """
    client, headers = streaming_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=headers["alice"]).json()
    events = stream_events(client, headers["alice"], conv["id"])

    done = next(e for e in events if e["event"] == "done")
    streamed = "".join(e["delta"] for e in events if e["event"] == "token")
    assert streamed == done["answer"]
    assert [e["event"] for e in events] == ["start", "token", "token", "done"]


@needs_web
def test_refused_stream_ships_the_suggestion_the_ui_renders(tmp_path: Path) -> None:
    """A refused stream must carry the rephrase suggestion ChatMessage shows."""
    client, headers = streaming_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=headers["alice"]).json()

    def refused(
        user_message: str, session: Session, use_web: bool = False
    ) -> Any:
        yield StreamRefused(rephrase_suggestion="Hãy hỏi lại với từ khóa khác.")

    state = cast(AppState, cast(FastAPI, client.app).state.coursemate)
    state.core.stream_answer = refused  # type: ignore[method-assign]
    events = stream_events(client, headers["alice"], conv["id"])

    assert [e["event"] for e in events] == ["refused"]
    assert events[0]["rephrase_suggestion"] == "Hãy hỏi lại với từ khóa khác."


@needs_web
def test_sync_chat_persists_the_exchange_the_sidebar_shows(tmp_path: Path) -> None:
    """POST /chat's turn must match TurnOut and land in the messages history."""
    client, headers = streaming_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=headers["alice"]).json()
    response = client.post(
        f"/api/conversations/{conv['id']}/chat",
        json={"message": "bảng băm là gì?"},
        headers=headers["alice"],
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert set(body) == set(TurnOut.model_fields)
    assert set(body["citations"][0]) == set(CitationOut.model_fields)
    assert set(body["citations"][0]["source"]) == set(SourceOut.model_fields)

    history = client.get(
        f"/api/conversations/{conv['id']}/messages", headers=headers["alice"]
    ).json()
    assert [turn["role"] for turn in history] == ["user", "assistant"]
    assert history[1]["text"] == body["text"], "history and response must agree"


@needs_web
def test_conversation_list_carries_every_field_the_sidebar_reads(tmp_path: Path) -> None:
    """ConversationMeta fields must survive into the sidebar's list payload."""
    client, _, _, headers = make_client(tmp_path)
    created = client.post("/api/conversations", json={}, headers=headers["alice"]).json()
    listed = client.get("/api/conversations", headers=headers["alice"]).json()

    assert len(listed) == 1
    assert set(listed[0]) == set(ConversationOut.model_fields)
    missing = required_fields(ts_interfaces(), "ConversationMeta") - set(listed[0])
    assert not missing, f"list payload lacks {sorted(missing)}"
    assert listed[0]["id"] == created["id"]


@needs_web
def test_upload_response_matches_attachment_listing(tmp_path: Path) -> None:
    """AttachmentsSection reads one shape from both upload and list."""
    client, _, _, headers = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=headers["alice"]).json()
    uploaded = client.post(
        "/api/attachments",
        data={"conversation_id": conv["id"]},
        files={"file": ("notes.md", b"# Chuong 1\nNoi dung A\n")},
        headers=headers["alice"],
    ).json()
    listed = client.get(
        "/api/attachments",
        params={"conversation_id": conv["id"]},
        headers=headers["alice"],
    ).json()

    assert set(uploaded) == set(AttachmentOut.model_fields)
    missing = required_fields(ts_interfaces(), "Attachment") - set(uploaded)
    assert not missing, f"upload payload lacks {sorted(missing)}"
    assert listed == [uploaded], "list must return exactly what upload returned"


@needs_web
def test_entity_bundle_shape_is_what_citation_cards_destructures(tmp_path: Path) -> None:
    """Citation cards resolve citations through the bundle; check its shape."""
    client, headers = streaming_client(tmp_path)
    bundle = client.get("/api/entities", headers=headers["alice"]).json()

    assert set(bundle) == set(EntitiesOut.model_fields)
    assert required_fields(ts_interfaces(), "EntityBundle") <= set(bundle)
    # The UI labels each card from entity.type, so it must survive the round trip.
    expected_type = {
        "departments": "department",
        "instructors": "instructor",
        "programs": "program",
        "terms": "term",
    }
    for key, label in expected_type.items():
        assert bundle[key], f"{key} must not be empty for this stub"
        for entity in bundle[key]:
            assert set(entity) == set(EntityOut.model_fields), (
                f"{key} entity omits fields the UI reads"
            )
            assert entity["type"] == label
            assert entity["id"] != ""
            assert isinstance(entity["detail"], dict)


@needs_web
def test_features_flag_drives_the_web_toggle(tmp_path: Path) -> None:
    """Sidebar renders the web toggle only when Features.has_web_search is true."""
    client, _, core, headers = make_client(tmp_path)
    body = client.get("/api/features", headers=headers["alice"]).json()

    assert set(body) == set(FeaturesOut.model_fields)
    assert body["has_web_search"] is False
    assert body["has_web_search"] == core.has_web_search
