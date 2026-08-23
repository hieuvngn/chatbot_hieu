from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from httpx2 import Response

from rag_core.attachments import MAX_FILES_PER_CONVERSATION
from tests.test_server_api import make_client


def _upload(
    client: TestClient,
    headers: dict[str, str],
    conv_id: str,
    name: str,
    content: str | bytes,
) -> Response:
    return client.post(
        "/api/attachments",
        data={"conversation_id": conv_id},
        files={"file": (name, content)},
        headers=headers,
    )


def test_upload_list_delete(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()

    uploaded = _upload(client, h["alice"], conv["id"], "notes.txt", "# Tieu de\nNoi dung")
    assert uploaded.status_code == 201
    meta = uploaded.json()
    assert meta["filename"] == "notes.txt"
    assert meta["chunk_count"] >= 1

    listed = client.get("/api/attachments", params={"conversation_id": conv["id"]}, headers=h["alice"]).json()
    assert [a["id"] for a in listed] == [meta["id"]]

    assert client.delete(f"/api/attachments/{meta['id']}", headers=h["alice"]).status_code == 204
    assert client.delete(f"/api/attachments/{meta['id']}", headers=h["alice"]).status_code == 404


def test_upload_limit_and_bad_type(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()
    for i in range(MAX_FILES_PER_CONVERSATION):
        assert _upload(client, h["alice"], conv["id"], f"f{i}.txt", b"noi dung").status_code == 201
    assert _upload(client, h["alice"], conv["id"], "extra.txt", b"noi dung").status_code == 400
    assert _upload(client, h["alice"], conv["id"], "x.exe", b"MZ").status_code == 400


def test_upload_ownership(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()
    stolen = _upload(client, h["bob"], conv["id"], "steal.txt", b"x")
    assert stolen.status_code == 404
    listed = client.get("/api/attachments", params={"conversation_id": conv["id"]}, headers=h["bob"])
    assert listed.status_code == 404
