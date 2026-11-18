import json
import tempfile
import unittest
from pathlib import Path

from generation import campaign_journal as j


class CampaignJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_chain_and_failed_draw_stop(self):
        j.record(self.out, "draw_started", {"operation_id": "atoms-0"})
        j.record(self.out, "draw_failed", {"operation_id": "atoms-0", "reason": "review"})
        self.assertTrue(j.is_stopped(self.out))
        self.assertEqual(len(j.audit(self.out)["events"]), 2)

    def test_unfinished_draw_is_not_silently_active(self):
        j.record(self.out, "draw_started", {"operation_id": "atoms-0"})
        with self.assertRaisesRegex(ValueError, "unfinished"):
            j.audit(self.out)
        self.assertTrue(j.is_stopped(self.out))

    def test_partial_line_and_chain_edit_fail_closed(self):
        j.record(self.out, "draw_started", {"operation_id": "atoms-0"})
        path = self.out / j.JOURNAL_NAME
        path.write_bytes(path.read_bytes() + b"{")
        with self.assertRaises(ValueError): j.audit(self.out, allow_active=True)
        path.write_bytes(path.read_bytes()[:-1])
        row = json.loads(path.read_text().strip()); row["payload"]["operation_id"] = "changed"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError): j.audit(self.out, allow_active=True)

    def test_atomic_artifact_is_idempotent_and_immutable(self):
        digest = j.atomic_write(self.out, "selected/a.json", b"abc")
        self.assertEqual(digest, j.sha256_bytes(b"abc"))
        self.assertEqual(j.atomic_write(self.out, "selected/a.json", b"abc"), digest)
        with self.assertRaises(FileExistsError): j.atomic_write(self.out, "selected/a.json", b"different")
        j.record(self.out, "draw_started", {"operation_id": "atoms-0", "artifacts": {"selected/a.json": digest}})
        (self.out / "selected/a.json").write_bytes(b"edited")
        with self.assertRaises(ValueError): j.audit(self.out, allow_active=True)

    def test_deleted_bound_artifact_is_detected(self):
        digest = j.atomic_write(self.out, "review.json", b"review")
        j.record(self.out, "draw_started", {"operation_id": "atoms-0", "artifacts": {"review.json": digest}})
        (self.out / "review.json").unlink()
        with self.assertRaises(ValueError): j.audit(self.out, allow_active=True)

    def test_intent_recovery_after_artifact_write_before_target(self):
        target = self.out / "intent.json"
        # Simulate the durable intent and artifact write, but crash before the
        # target event.  Recovery must use only the bytes in the intent.
        data = b"declared"
        payload = {"operation_id": "atoms-0"}
        rows = j.audit(self.out, allow_active=True)["events"]
        spec = {"intent_id": "i1", "target_event": "representation_locked",
                "target_payload": payload,
                "artifacts": {"intent.json": {"data": __import__('base64').b64encode(data).decode(), "sha256": j.sha256_bytes(data)}}}
        row = {"seq": 0, "event": "event_intent", "payload": spec,
               "previous_hash": "0" * 64}
        row["event_hash"] = j._event_hash(0, "event_intent", spec, row["previous_hash"])
        j._append_row(self.out, row)
        target.write_bytes(data)
        completed = j.recover_pending(self.out)
        self.assertEqual(completed["event"], "representation_locked")
        self.assertEqual(j.audit(self.out, allow_active=True)["events"][-1]["payload"]["intent_id"], "i1")

    def test_intent_conflicting_existing_artifact_fails(self):
        data = b"declared"; (self.out / "x").write_bytes(b"wrong")
        spec = {"intent_id": "i2", "target_event": "representation_locked", "target_payload": {},
                "artifacts": {"x": {"data": __import__('base64').b64encode(data).decode(), "sha256": j.sha256_bytes(data)}}}
        row = {"seq": 0, "event": "event_intent", "payload": spec, "previous_hash": "0" * 64}
        row["event_hash"] = j._event_hash(0, "event_intent", spec, row["previous_hash"]); j._append_row(self.out, row)
        with self.assertRaises(ValueError): j.recover_pending(self.out)


if __name__ == "__main__": unittest.main()
