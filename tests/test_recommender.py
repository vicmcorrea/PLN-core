from __future__ import annotations

import unittest

from pln_core.recommender import Song, recommend


def _song(label: str, valence: float, title: str = "x", artist: str = "y") -> Song:
    return Song(
        title=title,
        artist=artist,
        youtube_id="abc",
        sentiment=label,
        valence=valence,
        tags=(),
    )


class RecommenderTests(unittest.TestCase):
    def test_filters_by_label(self) -> None:
        catalog = (
            _song("positive", 0.8, "p1"),
            _song("negative", -0.6, "n1"),
            _song("neutral", 0.0, "u1"),
        )
        result = recommend("negative", -0.5, k=3, catalog=catalog)
        self.assertEqual([song.title for song in result], ["n1"])

    def test_ranks_by_valence_proximity(self) -> None:
        catalog = (
            _song("positive", 0.2, "low"),
            _song("positive", 0.95, "high"),
            _song("positive", 0.7, "mid"),
        )
        result = recommend("positive", 0.85, k=3, catalog=catalog)
        self.assertEqual([song.title for song in result], ["high", "mid", "low"])

    def test_clips_score_to_unit_interval(self) -> None:
        catalog = (
            _song("positive", 0.5, "softer"),
            _song("positive", 0.99, "stronger"),
        )
        result = recommend("positive", 12.0, k=1, catalog=catalog)
        self.assertEqual(result[0].title, "stronger")

    def test_returns_empty_when_no_matches(self) -> None:
        catalog = (_song("positive", 0.5, "p"),)
        result = recommend("negative", -0.9, k=3, catalog=catalog)
        self.assertEqual(result, ())

    def test_respects_k(self) -> None:
        catalog = tuple(_song("neutral", 0.1 * i, f"s{i}") for i in range(5))
        result = recommend("neutral", 0.0, k=2, catalog=catalog)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
