import unittest

from retainrec.models import PolicyClause, PolicyState, Relation
from retainrec.policy import policy_edits


class PolicyOperationTest(unittest.TestCase):
    def test_replace_is_one_operation(self) -> None:
        before_relation = Relation("genre", "drama", "horror")
        after_relation = Relation("genre", "comedy", "horror")
        before = PolicyState("u", (PolicyClause(before_relation, "support"),))
        after = PolicyState("u", (PolicyClause(after_relation, "requirement"),), version=1)
        edits = policy_edits(before, after)
        self.assertEqual(tuple(row.operation for row in edits), ("replace",))

    def test_scope_change_is_one_operation(self) -> None:
        before_relation = Relation("genre", "drama", "horror")
        after_relation = Relation(
            "genre",
            "drama",
            "horror",
            context_attribute="content_type",
            context_value="movie",
        )
        before = PolicyState("u", (PolicyClause(before_relation, "support"),))
        after = PolicyState("u", (PolicyClause(after_relation, "requirement"),), version=1)
        edits = policy_edits(before, after)
        self.assertEqual(tuple(row.operation for row in edits), ("scope",))


if __name__ == "__main__":
    unittest.main()
