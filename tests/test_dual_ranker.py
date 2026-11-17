import unittest

from generation.dual_ranker import dominates, rank_and_select


def row(i, family="f", novelty=0.5, feasibility=0.5, gain=0.5, metric="m", baseline="b", budget=1):
    return {"id": i, "problem_family": family,
            "discovery": {"novelty_estimate": novelty, "uncertainty": .1,
                          "evidence_status": "unresolved", "source_consistent": True},
            "execution": {"feasibility": feasibility, "expected_gain": gain,
                          "utility_uncertainty": .1, "metric": metric, "baseline_id": baseline, "budget": budget}}


class DualRankerTests(unittest.TestCase):
  def test_complete_supplied_numbers_do_not_promote_confidence(self):
    result=rank_and_select([row("a",novelty=.99,feasibility=.99,gain=100)],1)
    self.assertEqual(len(result.selected),1)
    self.assertFalse(result.selected[0].exploration)
    self.assertFalse(result.selected[0].promoted_confidence)

  def test_pareto_domination(self):
    self.assertTrue(dominates(row("a", novelty=.8, feasibility=.8, gain=.8), row("b")))
    self.assertFalse(dominates(row("b"), row("a")))


  def test_cross_metric_utility_is_incomparable(self):
    self.assertFalse(dominates(row("a", novelty=.8, feasibility=.8, gain=.9), row("b", novelty=.7, feasibility=.7, gain=.1, metric="other")))


  def test_unknown_is_exploration_not_zero(self):
    result = rank_and_select([row("known", novelty=.1, feasibility=.9), row("unknown", novelty=None, feasibility=None)], 1, exploration_quota=1)
    self.assertEqual(result.selected[0].id, "unknown")
    self.assertTrue(result.selected[0].exploration)
    self.assertFalse(result.selected[0].promoted_confidence)

  def test_zero_quota_does_not_select_unknown(self):
    result = rank_and_select([row("unknown", novelty=None, feasibility=None), row("known")], 2)
    self.assertEqual([s.id for s in result.selected], ["known"])

  def test_missing_utility_is_exploration_only(self):
    candidate = row("u"); candidate["execution"]["expected_gain"] = None
    result = rank_and_select([candidate, row("k")], 2, exploration_quota=1)
    self.assertTrue(result.selected[0].exploration)
    self.assertEqual(result.selected[0].id, "u")

  def test_numeric_bounds_and_count_type(self):
    with self.assertRaises(ValueError):
      rank_and_select([row("x", novelty=1.1)], 1)
    with self.assertRaises(ValueError):
      rank_and_select([row("x", feasibility=-.1)], 1)
    with self.assertRaises(ValueError):
      rank_and_select([row("x")], 1.0)

  def test_exploration_tags_are_quota_slots(self):
    result = rank_and_select([row("a", family="a"), row("b", family="b"), row("c", family="c")], 3, exploration_quota=2, seed=9)
    self.assertEqual(sum(s.exploration for s in result.selected), 2)
    self.assertTrue(all(s.reason == "exploration_quota" for s in result.selected[:2]))

  def test_seeded_family_order_is_repeatable_and_not_alphabetical(self):
    rows = [row("a", family="a"), row("b", family="b"), row("c", family="c")]
    x = [s.row["problem_family"] for s in rank_and_select(rows, 3, exploration_quota=3, seed=1).selected]
    y = [s.row["problem_family"] for s in rank_and_select(list(reversed(rows)), 3, exploration_quota=3, seed=1).selected]
    self.assertEqual(x, y)
    self.assertNotEqual(x, ["a", "b", "c"])


  def test_diversity_quota_covers_families(self):
    rows = [row("a", family="a"), row("b", family="b"), row("c", family="c")]
    selected = rank_and_select(rows, 2, exploration_quota=2).selected
    self.assertEqual({x.row["problem_family"] for x in selected}, {"a", "b"})


  def test_order_invariant_and_seeded_ties(self):
    rows = [row("a"), row("b"), row("c")]
    x = [s.id for s in rank_and_select(rows, 2, seed=4).selected]
    y = [s.id for s in rank_and_select(list(reversed(rows)), 2, seed=4).selected]
    self.assertEqual(x, y)


  def test_bad_nonfinite_rejected(self):
    r = row("x", novelty=float("nan"))
    with self.assertRaises(ValueError):
        rank_and_select([r], 1)


  def test_gates_invalid_and_inconsistent(self):
    bad = row("bad"); bad["known_invalid"] = True
    inconsistent = row("inc"); inconsistent["discovery"]["source_consistent"] = False
    result = rank_and_select([bad, inconsistent], 2)
    self.assertEqual(result.selected, ())
    self.assertEqual(result.rejected, {"bad": "known_invalid", "inc": "source_inconsistent"})


if __name__ == "__main__":
    unittest.main()

