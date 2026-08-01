import unittest

from financial_pragmatic_ai.analysis.segment_sampler import (
    select_representative_segments,
)


def make_segments(count):
    return [
        {
            "id": index,
            "speaker": "ANALYST" if index % 5 == 0 else "CEO",
            "text": (
                f"Segment {index}: revenue growth outlook and analyst question"
                if index in {4, 22, 51}
                else f"Segment {index}: routine business update"
            ),
        }
        for index in range(count)
    ]


class SegmentSamplerTests(unittest.TestCase):
    def test_selection_respects_budget_and_preserves_source_order(self):
        selected = select_representative_segments(make_segments(60), budget=12)

        self.assertEqual(len(selected), 12)
        self.assertEqual(
            [segment["source_index"] for segment in selected],
            sorted(segment["source_index"] for segment in selected),
        )

    def test_selection_covers_early_middle_and_late_regions(self):
        selected = select_representative_segments(make_segments(90), budget=12)
        source_indices = [segment["source_index"] for segment in selected]

        self.assertTrue(any(index < 30 for index in source_indices))
        self.assertTrue(any(30 <= index < 60 for index in source_indices))
        self.assertTrue(any(index >= 60 for index in source_indices))

    def test_selection_returns_all_segments_within_budget(self):
        selected = select_representative_segments(make_segments(4), budget=12)

        self.assertEqual([segment["id"] for segment in selected], [0, 1, 2, 3])
        self.assertEqual(
            [segment["source_index"] for segment in selected], [0, 1, 2, 3]
        )

    def test_selection_is_deterministic(self):
        segments = make_segments(60)

        first = select_representative_segments(segments, budget=12)
        second = select_representative_segments(segments, budget=12)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
