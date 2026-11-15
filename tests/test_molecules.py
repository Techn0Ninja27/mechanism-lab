import hashlib
import tempfile
import unittest
from pathlib import Path

from generation.molecules import MoleculeError, atom_record, molecule_record, molecule_id, promote


def atoms(order=("a", "b")):
    return [atom_record(x, "definition " + x, [{"source": "paper-" + x}]) for x in order]


class MoleculeTests(unittest.TestCase):
    def test_order_invariant_hash(self):
        a = atoms()
        kwargs = dict(relations=[{"source": "a", "target": "b", "type": "enables"}], mechanism="a enables b", assumptions=["a is available"], falsifiable_prediction="removing a lowers y")
        self.assertEqual(molecule_id(molecule_record(a, **kwargs)), molecule_id(molecule_record(list(reversed(a)), list(reversed(kwargs["relations"])), mechanism=kwargs["mechanism"], assumptions=kwargs["assumptions"], falsifiable_prediction=kwargs["falsifiable_prediction"])))

    def test_edge_changes_hash(self):
        a = atoms()
        x = molecule_record(a, [{"source": "a", "target": "b", "type": "enables"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        y = molecule_record(a, [{"source": "a", "target": "b", "type": "inhibits"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        self.assertNotEqual(x["id"], y["id"])

    def test_unknown_and_cycle_rejected(self):
        with self.assertRaises(MoleculeError): molecule_record(atoms(), [{"source": "a", "target": "z", "type": "x"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        with self.assertRaises(MoleculeError): molecule_record(atoms(("a", "b", "c")), [{"source": "a", "target": "b", "type": "x"}, {"source": "b", "target": "c", "type": "x"}, {"source": "c", "target": "a", "type": "x"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        with self.assertRaises(MoleculeError): molecule_record([], [], mechanism="m", assumptions=["x"], falsifiable_prediction="y")

    def test_promotion_requires_all_gates(self):
        m = molecule_record(atoms(), [{"source": "a", "target": "b", "type": "enables"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        with self.assertRaises(MoleculeError): promote(m, {"validation": True})
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "evidence.json"
            artifact.write_text("evidence")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            evidence = {key: {"passed": True, "artifact": str(artifact), "sha256": digest, "task_id": ("task-a" if key == "validation" else "task-b" if key == "independent_heldout_positive" else key)} for key in ("validation", "independent_heldout_positive", "ablation", "prior_art_review")}
            evidence["prior_art_review"]["human_adjudicated"] = True
            macro = promote(m, evidence)
            self.assertEqual(macro["type"], "macro_atom")
            self.assertFalse(macro["scientific_novelty_claim"])
            self.assertEqual(macro["decomposition"]["atom_ids"], ["a", "b"])
            self.assertEqual(macro["decomposition"]["atoms"], atoms())

    def test_promotion_rejects_false_dict_and_mutation(self):
        m = molecule_record(atoms(), [{"source": "a", "target": "b", "type": "enables"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "evidence.json"
            artifact.write_text("evidence")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            evidence = {key: {"passed": True, "artifact": str(artifact), "sha256": digest, "task_id": ("task-a" if key == "validation" else "task-b" if key == "independent_heldout_positive" else key)} for key in ("validation", "independent_heldout_positive", "ablation", "prior_art_review")}
            evidence["prior_art_review"]["human_adjudicated"] = True
            evidence["ablation"]["passed"] = False
            with self.assertRaises(MoleculeError): promote(m, evidence)
            evidence["ablation"]["passed"] = True
            m["mechanism"] = "mutated"
            with self.assertRaises(MoleculeError): promote(m, evidence)

    def test_promotion_checks_artifact_and_human_review(self):
        m = molecule_record(atoms(), [{"source": "a", "target": "b", "type": "enables"}], mechanism="m", assumptions=["x"], falsifiable_prediction="y")
        evidence = {key: {"passed": True, "artifact": "/does/not/exist", "sha256": "a" * 64, "task_id": ("task-a" if key == "validation" else "task-b" if key == "independent_heldout_positive" else key)} for key in ("validation", "independent_heldout_positive", "ablation", "prior_art_review")}
        with self.assertRaises(MoleculeError): promote(m, evidence)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "evidence.json"
            artifact.write_text("evidence")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            for item in evidence.values():
                item["artifact"] = str(artifact)
                item["sha256"] = digest
            with self.assertRaises(MoleculeError): promote(m, evidence)
            evidence["prior_art_review"]["human_adjudicated"] = True
            self.assertEqual(promote(m, evidence)["type"], "macro_atom")

    def test_rejects_string_assumptions_and_bad_provenance(self):
        with self.assertRaises(MoleculeError): molecule_record(atoms(), [], mechanism="m", assumptions="one", falsifiable_prediction="y")
        with self.assertRaises(MoleculeError): molecule_record(atoms(), [], mechanism="m", assumptions=["one"], falsifiable_prediction="y", provenance=["bad"])
        with self.assertRaises(MoleculeError): molecule_record(atoms(), [{"source": "a", "target": "b", "type": "x", "description": 3}], mechanism="m", assumptions=["one"], falsifiable_prediction="y")


if __name__ == "__main__": unittest.main()
