"""Durable local project/document workspace for the Alien Science app.

The workspace is deliberately separate from frozen engine results.  A project is
an explicit container for source material, working notes, reviews, exports, and
run outputs; it never mutates the experiment catalogue.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from pathlib import Path


MAX_FILE = 10 * 1024 * 1024
MAX_TOTAL = 40 * 1024 * 1024
MAX_FILES = 100
ALLOWED = {".md", ".txt", ".json", ".csv", ".pdf"}
CATEGORIES = {"notes", "reviews"}


def _synchronized(fn):
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    wrapped.__name__ = fn.__name__
    return wrapped


class WorkspaceError(ValueError):
    """An invalid workspace request."""


class Workspace:
    """Manage projects rooted at *root* using a small SQLite index."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()
        self.projects_root = self.root / "projects"
        self.db_path = self.root / "workspace.sqlite3"
        self._lock = threading.RLock()
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
          created_at REAL NOT NULL, path TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
          name TEXT NOT NULL, relative_path TEXT NOT NULL, size INTEGER NOT NULL,
          sha256 TEXT NOT NULL, kind TEXT NOT NULL, created_at REAL NOT NULL,
          UNIQUE(project_id, relative_path), UNIQUE(project_id, sha256)
        );
        CREATE INDEX IF NOT EXISTS docs_project ON documents(project_id);
        """)
        self._db.commit()

    def close(self):
        self._db.close()

    def _row(self, row):
        return dict(row) if row else None

    def _project(self, project_id):
        row = self._db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise WorkspaceError("unknown project")
        return row

    def _project_dir(self, project_id):
        row = self._project(project_id)
        path = (self.root / row["path"]).resolve()
        if self.projects_root not in path.parents:
            raise WorkspaceError("invalid project path")
        return path

    @staticmethod
    def _safe_name(name: str, *, extension_required=True):
        if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or ":" in name:
            raise WorkspaceError("invalid document name")
        p = Path(name)
        if p.is_absolute() or any(part in {"", ".", ".."} or part.startswith(".") for part in p.parts):
            raise WorkspaceError("absolute, traversal, and hidden paths are not allowed")
        if extension_required and p.suffix.lower() not in ALLOWED:
            raise WorkspaceError("unsupported document type")
        return p

    @staticmethod
    def _safe_project_name(name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise WorkspaceError("project name is required")
        return name.strip()

    @_synchronized
    def create_project(self, name: str, description: str = ""):
        name = self._safe_project_name(name)
        description = str(description or "").strip()[:2000]
        project_id = uuid.uuid4().hex[:16]
        rel = Path("projects") / project_id
        path = self.root / rel
        path.mkdir(parents=True, exist_ok=False)
        for folder in ("sources", "notes", "reviews", "exports", "runs"):
            (path / folder).mkdir()
        readme = path / "README.md"
        readme.write_text(
            f"# {name}\n\n{description}\n\n"
            "## Folder map\n\n"
            "- `sources/`: papers, datasets, PDFs, and other source material.\n"
            "- `notes/`: working notes and observations.\n"
            "- `reviews/`: adjudications, evidence checks, and decisions.\n"
            "- `runs/`: JSON references to local engine jobs; output files remain in results/webapp-jobs/.\n"
            "- `exports/`: generated ZIP handoffs; do not place source files here.\n",
            encoding="utf-8",
        )
        now = time.time()
        self._db.execute("INSERT INTO projects VALUES (?,?,?,?,?)", (project_id, name, description, now, str(rel)))
        self._db.commit()
        return self.get_project(project_id)

    @_synchronized
    def list_projects(self):
        return [self._row(r) for r in self._db.execute("SELECT * FROM projects ORDER BY created_at DESC")]

    @_synchronized
    def get_project(self, project_id):
        row = self._project(project_id)
        result = self._row(row)
        result["path"] = str(self._project_dir(project_id))
        result["document_count"] = self._db.execute("SELECT COUNT(*) FROM documents WHERE project_id=?", (project_id,)).fetchone()[0]
        return result

    @_synchronized
    def list_documents(self, project_id):
        self._project(project_id)
        return [self._row(r) for r in self._db.execute("SELECT * FROM documents WHERE project_id=? ORDER BY relative_path", (project_id,))]

    @_synchronized
    def import_documents(self, project_id, files):
        project_dir = self._project_dir(project_id)
        if not isinstance(files, list) or not files or len(files) > MAX_FILES:
            raise WorkspaceError(f"provide 1-{MAX_FILES} files")
        decoded = []
        seen_paths = set()
        total = 0
        for item in files:
            if not isinstance(item, dict):
                raise WorkspaceError("each file must be an object")
            p = self._safe_name(item.get("name"))
            if str(p) in seen_paths:
                raise WorkspaceError("duplicate path in batch")
            seen_paths.add(str(p))
            try: data = base64.b64decode(item.get("content_base64", ""), validate=True)
            except Exception as exc: raise WorkspaceError("content_base64 is invalid") from exc
            if len(data) > MAX_FILE: raise WorkspaceError(f"file exceeds {MAX_FILE} byte limit")
            total += len(data)
            if total > MAX_TOTAL: raise WorkspaceError(f"batch exceeds {MAX_TOTAL} byte limit")
            raw_target = project_dir / "sources" / p
            self._assert_no_symlink(raw_target, project_dir)
            target = raw_target.resolve()
            if project_dir not in target.parents: raise WorkspaceError("file escapes project")
            decoded.append((p, data, hashlib.sha256(data).hexdigest(), target))
        for p, data, digest, target in decoded:
            existing_path = self._db.execute("SELECT * FROM documents WHERE project_id=? AND relative_path=?", (project_id, str(Path("sources") / p))).fetchone()
            if existing_path and existing_path["sha256"] != digest:
                raise WorkspaceError(f"document path already exists with different content: {p}")
            self._assert_no_symlink(target, project_dir)
            if target.exists() and target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise WorkspaceError(f"document path already exists with different content: {p}")
        results = []
        for p, data, digest, target in decoded:
            old = self._db.execute("SELECT * FROM documents WHERE project_id=? AND sha256=?", (project_id, digest)).fetchone()
            if old:
                results.append(self._row(old)); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_symlink(): raise WorkspaceError("symlink destination rejected")
            target.write_bytes(data)
            doc_id = uuid.uuid4().hex[:16]
            kind = "pdf" if p.suffix.lower() == ".pdf" else "text"
            self._db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", (doc_id, project_id, p.name, str(Path("sources") / p), len(data), digest, kind, time.time()))
            results.append(self._row(self._db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()))
        self._db.commit()
        return results

    @_synchronized
    def read_document(self, project_id, document_id):
        project_dir = self._project_dir(project_id)
        row = self._db.execute("SELECT * FROM documents WHERE id=? AND project_id=?", (document_id, project_id)).fetchone()
        if not row: raise WorkspaceError("unknown document")
        path = self.document_path(project_id, document_id)
        if project_dir not in path.parents or not path.is_file(): raise WorkspaceError("document is unavailable")
        raw = path.read_bytes()
        result = self._row(row)
        if row["kind"] == "pdf": result.update({"binary": True, "content_base64": base64.b64encode(raw).decode()})
        else: result.update({"binary": False, "content": raw.decode("utf-8", errors="replace")})
        return result

    @_synchronized
    def save_note(self, project_id, title, content, category="notes"):
        if category not in CATEGORIES: raise WorkspaceError("category must be notes or reviews")
        p = self._safe_name(str(title).strip() + ".md", extension_required=True)
        project_dir = self._project_dir(project_id)
        raw_target = project_dir / category / p
        self._assert_no_symlink(raw_target, project_dir)
        target = raw_target.resolve()
        if project_dir not in target.parents: raise WorkspaceError("note escapes project")
        self._assert_no_symlink(target, project_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = str(content).encode("utf-8")
        if len(data) > MAX_FILE: raise WorkspaceError("note exceeds 10 MiB limit")
        digest = hashlib.sha256(data).hexdigest()
        old = self._db.execute("SELECT * FROM documents WHERE project_id=? AND sha256=?", (project_id, digest)).fetchone()
        if old: return self._row(old)
        old_path = self._db.execute("SELECT * FROM documents WHERE project_id=? AND relative_path=?", (project_id, str(Path(category) / p))).fetchone()
        if old_path and old_path["sha256"] != digest: raise WorkspaceError(f"note path already exists with different content: {p.name}")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest: raise WorkspaceError(f"note path already exists with different content: {p.name}")
        target.write_bytes(data); doc_id = uuid.uuid4().hex[:16]
        self._db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", (doc_id, project_id, p.name, str(Path(category) / p), len(data), digest, "text", time.time())); self._db.commit()
        return self._row(self._db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())

    @_synchronized
    def folder_map(self, project_id):
        project_dir = self._project_dir(project_id)
        descriptions = {"sources": "papers, datasets, PDFs, and source material", "notes": "working notes", "reviews": "evidence checks and decisions", "runs": "references to jobs; outputs live in results/webapp-jobs/", "exports": "generated ZIP handoffs"}
        return [{"name": f, "path": str((project_dir / f).resolve()), "description": d, "document_count": sum(1 for x in (project_dir / f).rglob("*") if x.is_file() and not x.is_symlink()) if (project_dir / f).exists() else 0} for f, d in descriptions.items()]

    @_synchronized
    def export_project(self, project_id):
        project = self.get_project(project_id); project_dir = self._project_dir(project_id)
        export = project_dir / "exports" / f"{project_id}-{time.time_ns()}.zip"
        manifest = {"project": project, "folders": self.folder_map(project_id), "documents": self.list_documents(project_id)}
        with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            for folder in ("sources", "notes", "reviews", "runs"):
                base = project_dir / folder
                self._assert_no_symlink(base, project_dir)
                for path in base.rglob("*"):
                    if path.is_symlink(): raise WorkspaceError("symlink in project cannot be exported")
                    if path.is_file(): z.write(path, Path(folder) / path.relative_to(base))
            z.write(project_dir / "README.md", "README.md")
        return {"path": str(export), "name": export.name, "size": export.stat().st_size, "project_id": project_id}

    @staticmethod
    def _assert_no_symlink(path, boundary):
        boundary = Path(boundary).resolve()
        path = Path(path)
        try:
            relative = path.relative_to(boundary)
        except ValueError as exc:
            raise WorkspaceError("path escapes project") from exc
        current = boundary
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise WorkspaceError("symlink path component rejected")

    @_synchronized
    def document_path(self, project_id, document_id):
        project_dir = self._project_dir(project_id)
        row = self._db.execute("SELECT relative_path FROM documents WHERE id=? AND project_id=?", (document_id, project_id)).fetchone()
        if not row:
            raise WorkspaceError("unknown document")
        raw_path = project_dir / row["relative_path"]
        self._assert_no_symlink(raw_path, project_dir)
        path = raw_path.resolve()
        if project_dir not in path.parents or not path.is_file():
            raise WorkspaceError("document is unavailable")
        self._assert_no_symlink(path, project_dir)
        return path

    @_synchronized
    def register_run(self, project_id, job_id, metadata=None):
        project_dir = self._project_dir(project_id)
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_id):
            raise WorkspaceError("invalid job id")
        target = project_dir / "runs" / f"{job_id}.json"
        if target.exists():
            raise WorkspaceError("run already registered")
        record = {"project_id": project_id, "job_id": job_id, "registered_at": time.time(), "metadata": metadata or {}}
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"job_id": job_id, "path": str(target), "metadata": record}
