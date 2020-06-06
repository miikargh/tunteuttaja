import pytest
import pandas as pd

from scripts.create_dataset import remove_blocklisted, load_blocklist


def test_remove_blocklisted_returns_correct_label_index():
    """ Function remove_blocklisted should return label_index that has
        blocklisted labels removed. """
    label_index = ["A", "B", "C", "D", "E"]
    blocklist = set([1, 2])
    df = pd.DataFrame([
        ("Testing testing 123", 1),
        ("Harry, you're a Wizard!", 2),
        ("All your base are belong to us", 4)
    ])
    df.columns = ["text", "label"]

    _, new_label_index = remove_blocklisted(df, blocklist, label_index)
    assert "B" not in new_label_index
    assert "C" not in new_label_index
    assert "A" in new_label_index
    assert "D" in new_label_index
    assert "E" in new_label_index


def test_remoce_blocklisted_returns_correct_df():
    """ Funtcion remove_blocklisted should return a new df with
        rows that have labels that are in blocklist removed
        and labels that correspond to the new label_list. """
    label_index = ["A", "B", "C", "D", "E"]
    blocklist = set([1, 2])
    df = pd.DataFrame([
        ("Testing testing 123", 1),
        ("Harry, you're a Wizard!", 2),
        ("All your base are belong to us", 4),
        ("Elementary, my dear Watson.", 0),
    ])
    df.columns = ["text", "label"]

    expected_df = pd.DataFrame([
        ("All your base are belong to us", 2),
        ("Elementary, my dear Watson.", 0),
    ])
    expected_df.columns = ["text", "label"]


    actual_df, _ = remove_blocklisted(df, blocklist, label_index)

    print(actual_df)
    print(expected_df)

    assert actual_df.equals(expected_df)
