import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_source_coherence import load_original, mapping_for, split_sets, main
from scripts.audit_source_coherence_baselines import scores


ROOT = Path(__file__).resolve().parents[1]


class SourceCoherenceTests(unittest.TestCase):
    def test_original_lineage_counts_and_text_hashes(self):
        data, papers, sets, units = load_original(ROOT / "papers/clusters.json")
        self.assertEqual(len(data["clusters"]), 273)
        self.assertEqual(len(units), 30580)
        self.assertEqual(len(papers), 13419)
        self.assertEqual(len(sets), 4503)
        self.assertTrue(all(len(u["text_sha256"]) == 64 for u in units))
        self.assertEqual(hashlib.sha256(ROOT.joinpath("papers/clusters.json").read_bytes()).hexdigest(),
                         "99b157f1baedba7908fca0236fb7dca23b3a49da44054ada305035036aa27013")

    def test_grouped_split_is_disjoint_and_reproducible(self):
        _, _, sets, _ = load_original(ROOT / "papers/clusters.json")
        a = split_sets(sets, 42); b = split_sets(sets, 42)
        self.assertEqual(a, b)
        self.assertEqual([len(x) for x in a], [3152, 675, 676])
        self.assertEqual(set(a[0]) & set(a[1]), set())
        self.assertEqual(set(a[0]) & set(a[2]), set())
        self.assertEqual(set(a[1]) & set(a[2]), set())

    def test_mapping_is_compact_and_source_sized(self):
        data, _, _, _ = load_original(ROOT / "papers/clusters.json")
        mapping = mapping_for(data)
        self.assertEqual(mapping.compute_vocab_size(), 274)
        self.assertEqual(mapping.bos_token_id, 273)
        self.assertEqual(mapping.eos_token_id, 273)
        self.assertEqual(mapping.cluster_ids_to_token_ids([0, 272]), [273, 0, 272, 273])

    def test_mapping_compacts_noncontiguous_cluster_ids(self):
        data = {"clusters": {"10": {}, "40": {}, "99": {}}}
        mapping = mapping_for(data)
        self.assertEqual(mapping.compute_vocab_size(), 4)
        self.assertEqual(mapping.cluster_ids_to_token_ids([10, 99]), [3, 0, 2, 3])

    def test_symmetric_pair_baseline_is_permutation_invariant(self):
        data, _, _, _ = load_original(ROOT / "papers/clusters.json")
        mapping = mapping_for(data)
        a = mapping.cluster_ids_to_token_ids([0, 1, 2])
        b = mapping.cluster_ids_to_token_ids([2, 0, 1])
        result_a = scores([(0, 1, 2)], [a], mapping)
        result_b = scores([(0, 1, 2)], [b], mapping)
        self.assertAlmostEqual(result_a["permutation_symmetric_conditional_pair_nll"],
                               result_b["permutation_symmetric_conditional_pair_nll"])

    def test_training_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"; out.mkdir(); (out / "keep").write_text("x")
            with self.assertRaises(SystemExit):
                main(["--output", str(out), "--max-steps", "1"])

    def test_baseline_audit_rejects_changed_cluster_source(self):
        from scripts.audit_source_coherence_baselines import main as audit
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text('{"clusters_sha256":"wrong"}')
            clusters = root / "clusters.json"; clusters.write_text('{}')
            with self.assertRaisesRegex(ValueError, "Cluster source"):
                audit(["--run", str(root), "--clusters", str(clusters)])
            self.assertFalse((root / "baseline-audit-conditional.json").exists())

    def test_training_archives_sources_and_loads_its_own_checkpoint(self):
        from generation.scoring import load_model
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); clusters = root / "clusters.json"
            atoms = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (0,1,2), (0,1,3), (0,2,3), (1,2,3)]
            clusters.write_text(json.dumps({"clusters": {str(i): {} for i in range(4)},
                "atom_assignments": [{"cluster": c, "paper_id": str(i), "text": str(c)} for i, row in enumerate(atoms) for c in row]}))
            out = root / "run"
            main(["--clusters", str(clusters), "--output", str(out), "--max-steps", "1", "--batch-size", "2"])
            manifest = json.loads((out / "manifest.json").read_text())
            for name, digest in manifest["source_hashes"].items():
                self.assertEqual(hashlib.sha256((out / "source_snapshot" / name).read_bytes()).hexdigest(), digest)
            frozen = json.loads((out / "freeze.json").read_text())
            self.assertEqual(hashlib.sha256((out / "checkpoint/model.pt").read_bytes()).hexdigest(), frozen["checkpoint_sha256"])
            model, mapping = load_model(out / "checkpoint")
            self.assertEqual(mapping.compute_vocab_size(), 5)


if __name__ == "__main__":
    unittest.main()
