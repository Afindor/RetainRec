import unittest

from retainrec.metrics import audit_transition, common_pass_urr
from retainrec.models import Item, PairCase, PolicyClause, PolicyState, Relation, Requirement
from retainrec.ranking import DeterministicRanker, relation_pass
from retainrec.selector import SelectionDecision
from retainrec.updater import DeterministicProposer, UpdateConfig, update_policy


class UpdaterMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.items = {
            "drama": Item("drama", {"genre": "drama"}),
            "horror": Item("horror", {"genre": "horror"}),
            "comedy": Item("comedy", {"genre": "comedy"}),
            "action": Item("action", {"genre": "action"}),
        }
        self.old = Relation("genre", "drama", "horror")
        self.new = Relation("genre", "comedy", "action")
        self.old_case = PairCase(
            "old",
            self.old,
            self.items["drama"],
            self.items["horror"],
            {},
            tuple(self.items),
        )
        self.new_case = PairCase(
            "new",
            self.new,
            self.items["comedy"],
            self.items["action"],
            {},
            tuple(self.items),
        )

    def test_bounded_update_preserves_selected_relation(self) -> None:
        ranker = DeterministicRanker(self.items)
        before = PolicyState("u", (PolicyClause(self.old, "support"),))
        requirement = Requirement("r", self.new, (self.new_case,), (self.new_case,))
        decision = SelectionDecision(self.old, 0.9, 0.8, True, "selected")
        config = UpdateConfig(
            2,
            4,
            0.82,
            0.05,
            16,
            0.5,
            0.5,
            {"add": 1.0, "delete": 1.0, "replace": 1.0, "scope": 1.0},
        )
        result = update_policy(
            state=before,
            requirements=(requirement,),
            decisions=(decision,),
            guard_cases={self.old.relation_id: (self.old_case,)},
            ranker=ranker,
            proposer=DeterministicProposer(),
            config=config,
        )
        self.assertFalse(result.rejected)
        self.assertEqual(len(result.proposals), 3)
        self.assertEqual([row.proposal_index for row in result.proposals], [0, 1, 2])
        self.assertTrue(relation_pass(ranker, result.state, (self.old_case,), 1.0))
        self.assertTrue(relation_pass(ranker, result.state, (self.new_case,), 1.0))
        metrics = audit_transition(
            before=before,
            after=result.state,
            audit_cases={self.old.relation_id: (self.old_case,)},
            requirements=(requirement,),
            selected_probabilities={self.old.relation_id: 0.9},
            ranker=ranker,
        )
        self.assertEqual(metrics.acquisition_rate, 1.0)
        self.assertEqual(metrics.update_regression_rate, 0.0)
        left, right, coverage = common_pass_urr(metrics, metrics)
        self.assertEqual((left, right, coverage), (0.0, 0.0, 1.0))

    def test_empty_pre_pass_denominator_remains_undefined(self) -> None:
        ranker = DeterministicRanker(self.items)
        empty = PolicyState("u", ())
        requirement = Requirement("r", self.new, (self.new_case,), (self.new_case,))
        metrics = audit_transition(
            before=empty,
            after=empty,
            audit_cases={self.old.relation_id: (self.old_case,)},
            requirements=(requirement,),
            selected_probabilities={},
            ranker=ranker,
        )
        self.assertIsNone(metrics.update_regression_rate)
        self.assertIsNone(metrics.obligation_generalization_rate)


if __name__ == "__main__":
    unittest.main()
