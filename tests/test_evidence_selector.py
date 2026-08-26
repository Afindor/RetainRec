import unittest

from retainrec.evidence import EvidenceSummary, aggregate_evidence, future_validity_label
from retainrec.models import Evidence, Relation
from retainrec.selector import SelectorSample, ValidityEstimator, choose_cost_ratio


def evidence(position: int, polarity: int, confidence: float = 1.0) -> Evidence:
    relation = Relation("genre", "drama", "horror")
    return Evidence(
        evidence_id=f"e{position}:{polarity}",
        user_id="u",
        position=position,
        relation=relation,
        polarity=polarity,
        confidence=confidence,
        preferred_item_id=f"p{position}",
        dispreferred_item_id=f"d{position}",
    )


class EvidenceSelectorTest(unittest.TestCase):
    def test_decay_and_future_label(self) -> None:
        summary = aggregate_evidence((evidence(1, 1), evidence(3, -1)), 3, 0.5)
        self.assertLess(summary.supporting, summary.contradicting)
        self.assertEqual(summary.distance_since_support, 2)
        self.assertEqual(future_validity_label((evidence(4, 1), evidence(5, 1)), 0.6, 2), 1)
        self.assertEqual(future_validity_label((evidence(4, -1), evidence(5, 1)), 0.6, 2), 0)
        self.assertIsNone(future_validity_label((evidence(4, 1),), 0.6, 2))

    def test_estimator_separates_validation_samples(self) -> None:
        samples = (
            SelectorSample(EvidenceSummary(4.0, 0.0, 0, 4), 1),
            SelectorSample(EvidenceSummary(3.0, 0.2, 1, 4), 1),
            SelectorSample(EvidenceSummary(0.0, 4.0, 5, 4), 0),
            SelectorSample(EvidenceSummary(0.2, 3.0, 4, 4), 0),
        )
        estimator = ValidityEstimator.fit(samples)
        self.assertGreater(estimator.predict(samples[0].summary), 0.5)
        self.assertLess(estimator.predict(samples[-1].summary), 0.5)
        ratio = choose_cost_ratio((0.9, 0.8, 0.2, 0.1), (1, 1, 0, 0), (0.5, 1.0, 2.0))
        self.assertIn(ratio, {0.5, 1.0, 2.0})


if __name__ == "__main__":
    unittest.main()
