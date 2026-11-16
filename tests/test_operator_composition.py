import unittest

from generation.operator_composition import (Operator, OperatorChain, OperatorCompositionError,
                                              OperatorRegistry, compose_chain)


class OperatorCompositionTests(unittest.TestCase):
    def setUp(self):
        self.registry = OperatorRegistry()
        self.registry.register(Operator("shift", "signed_ints", "signed_ints", lambda xs: [x + 1 for x in xs], lambda xs: [x - 1 for x in xs], version="1", source="test", contract_status="tested"))
        self.registry.register(Operator("pack", "signed_ints", "bytes", lambda xs: bytes((x % 256 for x in xs)), lambda b: list(b), version="1", source="test", contract_status="untested"))

    def test_execute_and_inverse_decode_in_reverse_order(self):
        chain = compose_chain(self.registry, ["shift", "pack"])
        self.assertEqual(chain.execute([1, 2]), bytes([2, 3]))
        self.assertEqual(chain.decode(bytes([2, 3])), [1, 2])

    def test_type_mismatch_and_runtime_contract(self):
        with self.assertRaises(OperatorCompositionError):
            compose_chain(self.registry, ["pack", "shift"])
        with self.assertRaises(OperatorCompositionError):
            compose_chain(self.registry, ["shift"]).execute((1, 2))
        with self.assertRaises(OperatorCompositionError):
            compose_chain(self.registry, ["shift"]).decode([1, True])

    def test_ids_preserve_distinct_same_type_operators(self):
        self.registry.register(Operator("shift_alt", "signed_ints", "signed_ints", lambda xs: [x + 2 for x in xs], lambda xs: [x - 2 for x in xs], source="test"))
        a = compose_chain(self.registry, ["shift"])
        b = compose_chain(self.registry, ["shift_alt"])
        self.assertNotEqual(a.content_id, b.content_id)
        self.assertEqual(len(self.registry.enumerate_chains("signed_ints", max_depth=1)), 3)

    def test_registry_limits_and_allowlist(self):
        self.assertLessEqual(len(self.registry.enumerate_chains("signed_ints", max_depth=5, max_candidates=2)), 2)
        with self.assertRaises(OperatorCompositionError):
            self.registry.enumerate_chains("signed_ints", allowed_registry_ids=["missing"])
        for bad in (True, 0, 9, 1.5):
            with self.assertRaises(OperatorCompositionError):
                self.registry.enumerate_chains("signed_ints", max_depth=bad)
        with self.assertRaises(OperatorCompositionError):
            self.registry.enumerate_chains("signed_ints", max_candidates=4097)

    def test_repeated_operator_occurrences_are_distinct_molecule_atoms(self):
        chain = compose_chain(self.registry, ["shift", "shift"])
        molecule = chain.to_molecule()
        self.assertEqual([a["id"] for a in molecule["atoms"]], ["shift#0", "shift#1"])
        self.assertEqual(molecule["relations"][0]["source"], "shift#0")

    def test_direct_chain_must_match_registry_definition(self):
        foreign = Operator("shift", "signed_ints", "signed_ints", lambda xs: xs, lambda xs: xs, source="foreign", implementation_digest="a" * 64)
        with self.assertRaises(OperatorCompositionError):
            OperatorChain((foreign,), self.registry)

    def test_molecule_is_conservative_and_content_addressed(self):
        molecule = compose_chain(self.registry, ["shift", "pack"]).to_molecule()
        self.assertEqual(molecule["type"], "molecule")
        self.assertEqual(molecule["status"], "combination")
        self.assertFalse(molecule["scientific_novelty_claim"])
        self.assertEqual([x["type"] for x in molecule["relations"]], ["composes"])


if __name__ == "__main__":
    unittest.main()
