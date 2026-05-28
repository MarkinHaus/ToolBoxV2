"""
VFS REST routes — global session view.

VFS is per-session. session_id is the chat_id.
"""
from __future__ import annotations

import io
import os
import zipfile


def register(app, ctx):
    isaa = ctx["isaa"]
    store = ctx["store"]
    uploads = ctx["uploads"]

    async def _get_vfs(session_id: str, agent_name: str | None = None):
        agent_name = agent_name or _default_agent_name(isaa, store, session_id)
        agent = await isaa.get_agent(agent_name)
        session = await agent.session_manager.get_or_create(session_id)
        return session.vfs

    @app.get("/api/vfs/tree")
    async def tree(request):
        qp = request.query_params
        session_id = _q(qp, "session_id")
        path = _q(qp, "path", "/")
        recursive = _q(qp, "recursive", "false").lower() in ("1", "true", "yes")
        if not session_id:
            return (400, {"error": "session_id required"})
        vfs = await _get_vfs(session_id)
        return vfs.ls(path, recursive=recursive)

    @app.get("/api/vfs/file")
    async def read_file(request):
        qp = request.query_params
        session_id = _q(qp, "session_id")
        path = _q(qp, "path")
        if not session_id or not path:
            return (400, {"error": "session_id and path required"})
        vfs = await _get_vfs(session_id)
        return vfs.read(path)

    @app.put("/api/vfs/file")
    async def write_file(request):
        body = request.json_data or {}
        session_id = body.get("session_id")
        path = body.get("path")
        content = body.get("content", "")
        if not session_id or not path:
            return (400, {"error": "session_id and path required"})
        vfs = await _get_vfs(session_id)
        return vfs.write(path, content)

    @app.post("/api/vfs/upload")
    async def upload(request):
        # Multipart parsing from request.form_data
        fd = request.form_data or {}
        session_id = _form_field(fd, "session_id")
        if not session_id:
            return (400, {"error": "session_id required"})
        target_dir = _form_field(fd, "path", "/uploads")

        file_field = fd.get("file")
        if not file_field:
            return (400, {"error": "file required"})

        # ParsedRequest exposes file as dict {filename, content_type, data: bytes} OR string content
        filename, data, content_type = _extract_file(file_field)
        if not data:
            return (400, {"error": "empty file"})

        # Persist temp file via upload_manager
        temp_dir = uploads.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / filename
        temp_path.write_bytes(data)

        meta = uploads.register_upload(
            user_id="markin",
            filename=filename,
            temp_path=str(temp_path),
            size=len(data),
            content_type=content_type or "application/octet-stream",
        )

        # Write into VFS at <target_dir>/<sanitized_name>
        vfs = await _get_vfs(session_id)
        vfs_path = _join(target_dir, meta.filename)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = f"[binary upload: {meta.filename} ({len(data)} bytes)]"
        vfs.create(vfs_path, text)

        return {"ok": True, "upload_id": meta.upload_id, "vfs_path": vfs_path}

    @app.get("/api/vfs/download")
    async def download(request):
        qp = request.query_params
        session_id = _q(qp, "session_id")
        path = _q(qp, "path")
        if not session_id or not path:
            return (400, {"error": "session_id and path required"})
        vfs = await _get_vfs(session_id)
        res = vfs.read(path)
        if not res.get("success"):
            return (404, {"error": res.get("error", "not found")})
        content = res.get("content", "")
        body = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        filename = os.path.basename(path) or "download.txt"
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
        return (200, headers, body)

    @app.get("/api/vfs/download_zip")
    async def download_zip(request):
        qp = request.query_params
        session_id = _q(qp, "session_id")
        path = _q(qp, "path", "/")
        if not session_id:
            return (400, {"error": "session_id required"})
        vfs = await _get_vfs(session_id)
        listing = vfs.ls(path, recursive=True)
        if not listing.get("success"):
            return (404, {"error": listing.get("error", "not found")})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in listing.get("contents", []):
                if item.get("type") != "file":
                    continue
                fp = item.get("path", "")
                if not fp:
                    continue
                rd = vfs.read(fp)
                if not rd.get("success"):
                    continue
                c = rd.get("content", "")
                rel = fp.lstrip("/")
                zf.writestr(rel, c.encode("utf-8") if isinstance(c, str) else bytes(c))

        body = buf.getvalue()
        filename = (path.strip("/").replace("/", "_") or "vfs") + ".zip"
        headers = {
            "Content-Type": "application/zip",
            "Content-Length": str(len(body)),
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
        return (200, headers, body)


def _q(qp, key: str, default: str = "") -> str:
    v = qp.get(key)
    if v is None:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return str(v)


def _form_field(fd, key: str, default: str = "") -> str:
    v = fd.get(key)
    if v is None:
        return default
    if isinstance(v, dict):  # could be file dict; coerce to filename
        return v.get("filename") or default
    return str(v)


def _extract_file(field):
    """Return (filename, bytes, content_type) from a form-file field."""
    if isinstance(field, dict):
        data = field.get("data") or field.get("content") or b""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return field.get("filename", "upload.bin"), data, field.get("content_type")
    # Worst case: just a string
    if isinstance(field, str):
        return "upload.txt", field.encode("utf-8"), "text/plain"
    if isinstance(field, (bytes, bytearray)):
        return "upload.bin", bytes(field), "application/octet-stream"
    return "upload.bin", b"", None


def _join(base: str, name: str) -> str:
    if not base.startswith("/"):
        base = "/" + base
    if not base.endswith("/"):
        base += "/"
    return base + name.lstrip("/")


def _default_agent_name(isaa, store, session_id: str) -> str:
    meta = store.get_meta(session_id) if store else None
    if meta and meta.agent:
        return meta.agent
    try:
        names = isaa.config.get("agents-name-list", [])
        if names:
            return names[0]
    except Exception:
        pass
    return "self"
