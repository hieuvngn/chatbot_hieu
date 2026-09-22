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


def test_get_content_endpoint_returns_sections(tmp_path: Path) -> None:
    # pytest có thể tái sử dụng tmp_path qua các lần chạy, xoá app.db cũ nếu có
    db_file = tmp_path / "app.db"
    if db_file.exists():
        db_file.unlink()
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()
    uploaded = _upload(
        client, h["alice"], conv["id"], "notes.md",
        b"# Chuong 1\nNoi dung A\n# Chuong 2\nNoi dung B\n",
    ).json()
    res = client.get(f"/api/attachments/{uploaded['id']}/content", headers=h["alice"])
    assert res.status_code == 200
    body = res.json()
    assert [s["chapter"] for s in body["sections"]] == ["Chuong 1", "Chuong 2"]
    assert body["sections"][0]["text"].startswith("Noi dung A")


def test_get_content_endpoint_ownership_and_not_found(tmp_path: Path) -> None:
    db_file = tmp_path / "app.db"
    if db_file.exists():
        db_file.unlink()
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()
    uploaded = _upload(client, h["alice"], conv["id"], "notes.txt", b"abc").json()
    # Bob không sở hữu conversation → 404
    assert client.get(
        f"/api/attachments/{uploaded['id']}/content", headers=h["bob"]
    ).status_code == 404
    # Attachment không tồn tại → 404
    assert client.get(
        "/api/attachments/nope/content", headers=h["alice"]
    ).status_code == 404
