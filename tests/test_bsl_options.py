"""Ensure that options for baseline estimates are taken into account."""


import numpy as np
import pytest  # type: ignore

from surprise import BaselineOnly
from surprise.model_selection import cross_validate


def test_method_field(u1_ml100k, pkf):
    """Ensure the method field is taken into account."""

    bsl_options = {"method": "als"}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    bsl_options = {"method": "sgd"}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    assert not np.array_equal(rmse_als, rmse_sgd)

    with pytest.raises(ValueError):
        bsl_options = {"method": "wrong_name"}
        algo = BaselineOnly(bsl_options=bsl_options)
        cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]


def test_als_n_epochs_field(u1_ml100k, pkf):
    """Ensure the n_epochs field is taken into account."""

    bsl_options = {
        "method": "als",
        "n_epochs": 1,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_1 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    bsl_options = {
        "method": "als",
        "n_epochs": 5,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_5 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    assert not np.array_equal(rmse_als_n_epochs_1, rmse_als_n_epochs_5)


def test_als_reg_u_field(u1_ml100k, pkf):
    """Ensure the reg_u field is taken into account."""

    bsl_options = {
        "method": "als",
        "reg_u": 0,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_0 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    bsl_options = {
        "method": "als",
        "reg_u": 10,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_10 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    assert not np.array_equal(rmse_als_regu_0, rmse_als_regu_10)


def test_als_reg_i_field(u1_ml100k, pkf):
    """Ensure the reg_i field is taken into account."""

    bsl_options = {
        "method": "als",
        "reg_i": 0,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_0 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    bsl_options = {
        "method": "als",
        "reg_i": 10,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_10 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    assert not np.array_equal(rmse_als_regi_0, rmse_als_regi_10)


def test_sgd_n_epoch_field(u1_ml100k, pkf):
    """Ensure the n_epoch field is taken into account."""

    bsl_options = {
        "method": "sgd",
        "n_epochs": 15,
    }
    algo = BaselineOnly(bsl_options=bsl_options)

    # Verify the parameter is stored correctly
    assert algo.bsl_options["n_epochs"] == 15

    # Test that the algorithm runs without error
    trainset, testset = next(pkf.split(u1_ml100k))
    algo.fit(trainset)
    predictions = algo.test(testset)
    assert len(predictions) > 0


def test_sgd_learning_rate_field(u1_ml100k, pkf):
    """Ensure the learning_rate field is taken into account."""

    bsl_options = {
        "method": "sgd",
        "n_epochs": 5,
        "learning_rate": 0.123,
    }
    algo = BaselineOnly(bsl_options=bsl_options)

    # Verify the parameter is stored correctly
    assert algo.bsl_options["learning_rate"] == 0.123

    # Test that the algorithm runs without error
    trainset, testset = next(pkf.split(u1_ml100k))
    algo.fit(trainset)
    predictions = algo.test(testset)
    assert len(predictions) > 0


def test_sgd_reg_field(u1_ml100k, pkf):
    """Ensure the reg field is taken into account."""

    bsl_options = {
        "method": "sgd",
        "n_epochs": 5,
        "reg": 0.456,
    }
    algo = BaselineOnly(bsl_options=bsl_options)

    # Verify the parameter is stored correctly
    assert algo.bsl_options["reg"] == 0.456

    # Test that the algorithm runs without error
    trainset, testset = next(pkf.split(u1_ml100k))
    algo.fit(trainset)
    predictions = algo.test(testset)
    assert len(predictions) > 0
