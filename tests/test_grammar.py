import unittest

from retainrec.grammar import GrammarConfig, RelationGrammar, canonical_pair_cases
from retainrec.models import Item


class GrammarTest(unittest.TestCase):
    def test_frozen_families_and_canonical_pairing(self) -> None:
        grammar = RelationGrammar(GrammarConfig(("genre",), ("region",)))
        preferred = Item("p", {"genre": "drama"})
        dispreferred = Item("d", {"genre": "horror"})
        relations = grammar.enumerate(preferred, dispreferred, {"region": "kr"})
        self.assertEqual(
            {relation.family for relation in relations},
            {"pairwise_preference", "contextual_override"},
        )
        rows = grammar.ground_event(
            user_id="u",
            event_id="e",
            position=1,
            preferred=preferred,
            dispreferred=dispreferred,
            context={"region": "kr"},
            confidence=1.0,
        )
        cases = canonical_pair_cases(rows, {"p": preferred, "d": dispreferred}, ("p", "d"), 2, "s")
        self.assertTrue(cases)
        for relation_cases in cases.values():
            self.assertLessEqual(len(relation_cases), 1)


if __name__ == "__main__":
    unittest.main()
