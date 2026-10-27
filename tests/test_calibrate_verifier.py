"""Offline checks for notebooks/calibrate_verifier.py (no Gemini calls)."""
from collections import Counter

from notebooks.calibrate_verifier import build_pairs, sample_balanced
from notebooks.eval_harness import load_medaesqa, verifier_agreement


def test_pairs():
    pairs = build_pairs(load_medaesqa())
    assert pairs
    assert {p["relation"] for p in pairs} <= {"supporting", "neutral", "contradicting"}
    assert all(p["evidence"] and p["sentence"] for p in pairs)
    assert not any("[" in p["sentence"] for p in pairs), "citations should be stripped"
    assert len({p["id"] for p in pairs}) == len(pairs)


def test_sample_is_balanced_and_repeatable():
    pairs = build_pairs(load_medaesqa())
    a, b = sample_balanced(pairs, 100), sample_balanced(pairs, 100)
    assert Counter(p["human_label"] for p in a) == {1: 50, 0: 50}
    assert [p["id"] for p in a] == [p["id"] for p in b]


def test_kappa_by_hand():
    # 10 items: agree on 8 (5 both-1, 3 both-0), disagree on 2
    human    = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    verifier = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    # observed agreement = 8/10 = 0.8
    # chance agreement = P(both 1) + P(both 0) = 0.6*0.6 + 0.4*0.4 = 0.52
    # kappa = (0.8 - 0.52) / (1 - 0.52) = 0.5833
    stats = verifier_agreement(human, verifier)
    assert stats["accuracy"] == 0.8
    assert abs(stats["cohen_kappa"] - 0.5833) < 1e-3


if __name__ == "__main__":
    test_pairs()
    test_sample_is_balanced_and_repeatable()
    test_kappa_by_hand()
    print("All calibration tests passed.")