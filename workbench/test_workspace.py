import base64
import tempfile
import unittest
import zipfile
from pathlib import Path

from workspace import Workspace, WorkspaceError


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)
        self.project = self.ws.create_project("Watermark study", "A small reproducible study")

    def tearDown(self):
        self.ws.close(); self.tmp.cleanup()

    def test_project_has_explicit_folders(self):
        paths = {x["name"] for x in self.ws.folder_map(self.project["id"])}
        self.assertEqual(paths, {"sources", "notes", "reviews", "runs", "exports"})
        self.assertTrue((Path(self.project["path"]) / "README.md").exists())

    def test_import_read_and_deduplicate(self):
        payload = base64.b64encode(b"hello world").decode()
        first = self.ws.import_documents(self.project["id"], [{"name": "papers/one.txt", "content_base64": payload}])
        second = self.ws.import_documents(self.project["id"], [{"name": "papers/copy.txt", "content_base64": payload}])
        self.assertEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(self.ws.read_document(self.project["id"], first[0]["id"])["content"], "hello world")

    def test_python_source_is_imported_as_text_never_executed(self):
        source = "raise RuntimeError('source must never execute')\n"
        doc = self.ws.import_documents(self.project['id'], [{'name':'method.py', 'content_base64':base64.b64encode(source.encode()).decode()}])[0]
        self.assertEqual(self.ws.read_document(self.project['id'],doc['id'])['content'],source)
        self.assertEqual(doc['kind'],'text')

    def test_pdf_is_binary(self):
        data = base64.b64encode(b"%PDF fake").decode()
        doc = self.ws.import_documents(self.project["id"], [{"name": "paper.pdf", "content_base64": data}])[0]
        out = self.ws.read_document(self.project["id"], doc["id"])
        self.assertTrue(out["binary"]); self.assertIn("content_base64", out)

    def test_rejects_unsafe_or_unsupported_files(self):
        encoded = base64.b64encode(b"x").decode()
        for name in ("../escape.txt", ".secret.txt", "/tmp/a.txt", "a.exe"):
            with self.subTest(name=name), self.assertRaises(WorkspaceError):
                self.ws.import_documents(self.project["id"], [{"name": name, "content_base64": encoded}])

    def test_conflicting_same_path_does_not_corrupt_index(self):
        a = base64.b64encode(b"one").decode()
        b = base64.b64encode(b"two").decode()
        self.ws.import_documents(self.project["id"], [{"name": "same.txt", "content_base64": a}])
        with self.assertRaises(WorkspaceError):
            self.ws.import_documents(self.project["id"], [{"name": "same.txt", "content_base64": b}])
        docs = self.ws.list_documents(self.project["id"])
        self.assertEqual(len(docs), 1)
        self.assertEqual(self.ws.read_document(self.project["id"], docs[0]["id"])["content"], "one")

    def test_document_path_and_run_registration(self):
        payload = base64.b64encode(b"source").decode()
        doc = self.ws.import_documents(self.project["id"], [{"name": "source.txt", "content_base64": payload}])[0]
        self.assertTrue(self.ws.document_path(self.project["id"], doc["id"]).is_file())
        run = self.ws.register_run(self.project["id"], "job_123", {"seed": 42})
        self.assertTrue(Path(run["path"]).is_file())

    def test_symlink_component_cannot_redirect_import(self):
        project = Path(self.project['path'])
        (project / 'sources' / 'redirect').symlink_to(project / 'notes', target_is_directory=True)
        with self.assertRaises(WorkspaceError):
            self.ws.import_documents(self.project['id'], [{'name': 'redirect/file.txt', 'content_base64': base64.b64encode(b'blocked').decode()}])
        self.assertFalse((project / 'notes/file.txt').exists())

    def test_export_rejects_external_symlink(self):
        project = Path(self.project['path'])
        outside = Path(self.tmp.name) / 'outside.txt'
        outside.write_text('must not be exported')
        (project / 'sources/leak.txt').symlink_to(outside)
        with self.assertRaises(WorkspaceError): self.ws.export_project(self.project['id'])

    def test_notes_reviews_and_export(self):
        self.ws.save_note(self.project["id"], "first pass", "Keep the baseline fixed", "notes")
        self.ws.save_note(self.project["id"], "decision", "Hold for provenance", "reviews")
        result = self.ws.export_project(self.project["id"])
        with zipfile.ZipFile(result["path"]) as z:
            names = set(z.namelist())
            self.assertIn("manifest.json", names)
            self.assertIn("notes/first pass.md", names)
            self.assertIn("reviews/decision.md", names)
            self.assertFalse(any(n.startswith("exports/") for n in names))


if __name__ == "__main__":
    unittest.main()
