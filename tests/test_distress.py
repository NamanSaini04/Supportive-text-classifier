"""Tests for the safety layer. This is the module that MUST work.

Run:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.distress import check_distress  # noqa: E402


def test_obvious_distress_triggers():
    assert check_distress("i don't want to be here anymore")
    assert check_distress("I can't take it anymore")
    assert check_distress("sometimes I just want to disappear")
    assert check_distress("everyone would be better off without me")


def test_neutral_does_not_trigger():
    assert not check_distress("i'm so tired of this rainy weather")
    assert not check_distress("work was stressful but I'm okay")
    assert not check_distress("")


def test_distress_runs_on_raw_text_with_punctuation():
    # Critical: must catch even with casing / punctuation noise
    assert check_distress("I DON'T want to live...")
