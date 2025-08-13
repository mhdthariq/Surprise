"""
Tests for rating_scale parameter in dataset creation methods.

This module tests the new functionality where rating_scale can be specified
directly in Dataset.load_from_file(), Dataset.load_from_folds(), and
Dataset.load_from_df() methods, with deprecation warnings for the old way.
"""

import os
import tempfile
import warnings

import pandas as pd

from surprise import Dataset, Reader


def test_rating_scale_in_load_from_df():
    """Test rating_scale parameter in Dataset.load_from_df()."""

    # Create test data
    ratings_dict = {
        "userID": [1, 1, 2, 2, 3],
        "itemID": [1, 2, 1, 3, 2],
        "rating": [4.0, 3.5, 2.0, 5.0, 1.5],
    }
    df = pd.DataFrame(ratings_dict)

    # Test: rating_scale specified in dataset creation (new way)
    reader = Reader()  # No rating_scale specified here
    data = Dataset.load_from_df(df, reader, rating_scale=(1, 5))
    trainset = data.build_full_trainset()
    assert trainset.rating_scale == (1, 5)

    # Test: rating_scale specified only in reader (old way, should show deprecation warning)
    reader_with_scale = Reader(rating_scale=(1, 10))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data2 = Dataset.load_from_df(df, reader_with_scale)
        trainset2 = data2.build_full_trainset()

        # Check that deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "rating_scale in Reader is deprecated" in str(w[0].message)

    assert trainset2.rating_scale == (1, 10)

    # Test: rating_scale specified in both (dataset should take precedence)
    reader_with_scale = Reader(rating_scale=(1, 10))
    data3 = Dataset.load_from_df(df, reader_with_scale, rating_scale=(0, 5))
    trainset3 = data3.build_full_trainset()
    assert trainset3.rating_scale == (0, 5)


def test_rating_scale_in_load_from_file():
    """Test rating_scale parameter in Dataset.load_from_file()."""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("1 1 4.0\n")
        f.write("1 2 3.5\n")
        f.write("2 1 2.0\n")
        f.write("2 3 5.0\n")
        temp_file = f.name

    try:
        # Test: rating_scale specified in dataset creation
        reader = Reader(line_format='user item rating', sep=' ')
        data = Dataset.load_from_file(temp_file, reader, rating_scale=(1, 5))
        trainset = data.build_full_trainset()
        assert trainset.rating_scale == (1, 5)

        # Test: rating_scale precedence (dataset over reader)
        reader_with_scale = Reader(line_format='user item rating', sep=' ', rating_scale=(1, 10))
        data2 = Dataset.load_from_file(temp_file, reader_with_scale, rating_scale=(0, 5))
        trainset2 = data2.build_full_trainset()
        assert trainset2.rating_scale == (0, 5)

    finally:
        os.unlink(temp_file)


def test_rating_scale_in_load_from_folds():
    """Test rating_scale parameter in Dataset.load_from_folds()."""

    # Create temporary train and test files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_train.txt') as f_train:
        f_train.write("1 1 4.0\n")
        f_train.write("1 2 3.5\n")
        train_file = f_train.name

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_test.txt') as f_test:
        f_test.write("2 1 2.0\n")
        f_test.write("2 3 5.0\n")
        test_file = f_test.name

    try:
        # Test: rating_scale specified in dataset creation
        reader = Reader(line_format='user item rating', sep=' ')
        folds_files = [(train_file, test_file)]
        data = Dataset.load_from_folds(folds_files, reader, rating_scale=(1, 5))

        # Since load_from_folds creates a DatasetUserFolds, we need to access it differently
        # We'll create a trainset by reading the train file
        raw_trainset = data.read_ratings(train_file)
        trainset = data.construct_trainset(raw_trainset)
        assert trainset.rating_scale == (1, 5)

        # Test: rating_scale precedence (dataset over reader)
        reader_with_scale = Reader(line_format='user item rating', sep=' ', rating_scale=(1, 10))
        data2 = Dataset.load_from_folds(folds_files, reader_with_scale, rating_scale=(0, 5))
        raw_trainset2 = data2.read_ratings(train_file)
        trainset2 = data2.construct_trainset(raw_trainset2)
        assert trainset2.rating_scale == (0, 5)

    finally:
        os.unlink(train_file)
        os.unlink(test_file)


def test_rating_scale_deprecation_warning_details():
    """Test that deprecation warning contains appropriate details."""

    ratings_dict = {
        "userID": [1, 2],
        "itemID": [1, 2],
        "rating": [4.0, 3.0],
    }
    df = pd.DataFrame(ratings_dict)
    reader_with_scale = Reader(rating_scale=(1, 10))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = Dataset.load_from_df(df, reader_with_scale)
        data.build_full_trainset()

        assert len(w) == 1
        warning = w[0]
        assert issubclass(warning.category, DeprecationWarning)
        message = str(warning.message)

        # Check that warning message mentions all three methods
        assert "Dataset.load_from_file()" in message
        assert "Dataset.load_from_folds()" in message
        assert "Dataset.load_from_df()" in message


def test_no_rating_scale_specified():
    """Test that Reader's default rating_scale is used when none specified."""

    ratings_dict = {
        "userID": [1, 2],
        "itemID": [1, 2],
        "rating": [4.0, 3.0],
    }
    df = pd.DataFrame(ratings_dict)

    # Reader with default rating_scale (1, 5)
    reader = Reader()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = Dataset.load_from_df(df, reader)
        trainset = data.build_full_trainset()

        # Should show deprecation warning since rating_scale comes from Reader
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Should use Reader's default rating_scale
    assert trainset.rating_scale == (1, 5)


def test_rating_scale_none_in_dataset():
    """Test behavior when rating_scale=None is explicitly passed to dataset methods."""

    ratings_dict = {
        "userID": [1, 2],
        "itemID": [1, 2],
        "rating": [4.0, 3.0],
    }
    df = pd.DataFrame(ratings_dict)
    reader_with_scale = Reader(rating_scale=(2, 10))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Explicitly pass None as rating_scale to dataset
        data = Dataset.load_from_df(df, reader_with_scale, rating_scale=None)
        trainset = data.build_full_trainset()

        # Should show deprecation warning since falling back to Reader
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Should use Reader's rating_scale since dataset rating_scale is None
    assert trainset.rating_scale == (2, 10)
