import unittest

from generation.atom_graph import edges, evaluate, ranking, score_pairs, split_sets, tfidf_vectors


class AtomGraphTests(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self):
        sets = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (0, 4)]
        a = split_sets(sets, 73031)
        b = split_sets(sets, 73031)
        self.assertEqual(a, b)
        self.assertFalse(set(a[0]) & set(a[1]))
        self.assertFalse(set(a[1]) & set(a[2]))

    def test_edges_are_undirected_pair_counts(self):
        self.assertEqual(edges([(0, 1, 2), (0, 1)]), {(0, 1): 2, (0, 2): 1, (1, 2): 1})

    def test_scores_exclude_observed_edges_and_are_deterministic(self):
        defs = {i: f"shared token {i}" for i in range(4)}
        train = edges([(0, 1), (1, 2)])
        scored = score_pairs([0, 1, 2, 3], train, "common_neighbors", defs)
        self.assertNotIn((0, 1), scored)
        self.assertEqual(ranking(scored), ranking(score_pairs([0, 1, 2, 3], train, "common_neighbors", defs)))

    def test_tfidf_is_finite_and_evaluation_counts_hits(self):
        defs = {0: "alpha beta", 1: "alpha gamma", 2: "delta epsilon"}
        vectors = tfidf_vectors(defs)
        self.assertTrue(all(v == v and abs(v) < 100 for d in vectors.values() for v in d.values()))
        self.assertEqual(evaluate([(0, 1), (0, 2)], {(0, 2)}, (1, 2))["hits@1"], 0)


if __name__ == "__main__":
    unittest.main()
