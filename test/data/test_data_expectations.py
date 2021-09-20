import os
import sys
from pathlib import Path

import great_expectations as ge
import pandas as pd
import pytest

# TODO: code smell of project estructure
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from src.config.config import Features, Paths


@pytest.fixture
def df():
    pandas_df = pd.read_csv(f"{Paths.RAW_DATA_FILE}")
    df = ge.from_pandas(pandas_df)
    return df

def test_expected_columns(df):
    # Presence of features
    expected_columns = Features.RAW_INPUT_FEATURES
    results = df.expect_table_columns_to_match_set(column_set=expected_columns, exact_match=True)
    assert(results["success"])

def test_duplicates_dates(df):
    # Unique
    results = df.expect_column_values_to_be_unique(column=f"{Features.DATE_FEATURE_NAME}")
    assert(results["success"])

def test_null_close(df):
    # No null values
    results = df.expect_column_values_to_not_be_null(column="close")
    assert(results["success"])

def test_rows(df):
    results = df.expect_table_row_count_to_be_between(min_value=20, max_value=1000000000)
    assert(results["success"])

def test_volume(df):
    result = df.expect_column_values_to_be_between(column="volume", min_value=0)
    assert(result["success"])

def test_types(df):
    # Expectation suite
    df.expect_column_values_to_be_of_type(column="open", type_="float")
    df.expect_column_values_to_be_of_type(column="close", type_="float")
    df.expect_column_values_to_be_of_type(column="volume", type_="int")

    expectation_suite = df.get_expectation_suite()
    results = df.validate(expectation_suite=expectation_suite, only_return_failures=True)
    assert(results["success"])
