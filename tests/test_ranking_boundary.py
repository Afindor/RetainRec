import sys
import unittest

from retainrec.models import Item, PairCase, PolicyState, Relation
from retainrec.ranking import CommandRanker, pair_pass


class RankingBoundaryTest(unittest.TestCase):
    def test_serving_request_withholds_pair_label(self) -> None:
        preferred = Item("a", {"genre": "drama"})
        dispreferred = Item("b", {"genre": "horror"})
        relation = Relation("genre", "drama", "horror")
        case = PairCase("case", relation, preferred, dispreferred, {}, ("a", "b"))
        program = (
            "import json,sys;"
            "p=json.load(sys.stdin);c=p['case'];"
            "assert 'preferred_item' not in c and 'dispreferred_item' not in c;"
            "json.dump(c['candidates'],sys.stdout)"
        )
        ranker = CommandRanker((sys.executable, "-c", program))
        self.assertEqual(pair_pass(ranker, PolicyState("u", ()), case), 1)


if __name__ == "__main__":
    unittest.main()
