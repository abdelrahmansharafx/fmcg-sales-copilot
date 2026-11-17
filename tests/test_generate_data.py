"""Sanity checks for the synthetic data generator."""

import numpy as np
import pytest

from scripts import generate_data as gd


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(1)
    outlets = gd.make_outlets(rng)
    skus = gd.make_skus()
    promos = gd.make_promotions(rng, skus)
    sales = gd.build_sales(rng, outlets, skus, promos)
    return outlets, skus, promos, sales


def test_dimension_sizes(world):
    outlets, skus, promos, _ = world
    assert len(outlets) == 300
    assert len(skus) == 25
    assert set(outlets["region"]) == set(gd.REGIONS)
    assert set(promos["discount_pct"]) <= set(gd.PROMO_DISCOUNTS)


def test_promotions_follow_the_policy(world):
    _, _, promos, _ = world
    for _, group in promos.groupby("sku_id"):
        group = group.sort_values("start_week")
        gaps = (group["start_week"].shift(-1) - group["end_week"]).dropna().dt.days // 7
        assert (gaps > gd.MIN_WEEKS_BETWEEN_PROMOS).all()
    lengths = (promos["end_week"] - promos["start_week"]).dt.days // 7 + 1
    assert lengths.between(1, 3).all()


def test_sales_rows_are_valid(world):
    outlets, _, _, sales = world
    assert not sales.duplicated(["week_start", "outlet_id", "sku_id"]).any()
    assert (sales["units"] >= 0).all()
    assert (sales["unit_price_egp"] > 0).all()
    # No sales before an outlet opened.
    merged = sales.merge(outlets[["outlet_id", "opened_date"]], on="outlet_id")
    assert (merged["week_start"] >= merged["opened_date"] - np.timedelta64(6, "D")).all()


def test_promo_weeks_sell_more_and_cost_less(world):
    _, _, _, sales = world
    sold = sales[sales["units"] > 0]
    promo = sold[sold["promo_id"].notna()]
    normal = sold[sold["promo_id"].isna()]
    assert promo["units"].mean() > normal["units"].mean()


def test_same_seed_gives_same_data():
    def build(seed: int):
        rng = np.random.default_rng(seed)
        outlets = gd.make_outlets(rng)
        skus = gd.make_skus()
        promos = gd.make_promotions(rng, skus)
        return gd.build_sales(rng, outlets, skus, promos)

    assert build(5).equals(build(5))
    assert not build(5).equals(build(6))


def test_expected_units_are_the_noise_free_mean(world):
    _, _, _, sales = world
    assert (sales["expected_units"] > 0).all()
    # Noise averages out; stock-outs remove a few %, so actual units sit a bit below the mean.
    ratio = sales["units"].sum() / sales["expected_units"].sum()
    assert 0.9 < ratio < 1.02
