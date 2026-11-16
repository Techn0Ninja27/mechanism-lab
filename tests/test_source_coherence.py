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

    def test_future_checkpoint_packages_its_mapping(self):
        source = (ROOT / "scripts/train_source_coherence.py").read_text()
        self.assertIn('(ckpt / "token_mapping.json").write_text(mapping_json)', source)


if __name__ == "__main__":
    unittest.main()
