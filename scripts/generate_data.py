"""Generate a seeded, synthetic FMCG sales dataset for Egypt and load it into SQLite.

Usage:
    uv run python scripts/generate_data.py            # writes data/sales.db
    uv run python scripts/generate_data.py --seed 7   # different (but reproducible) data

The data is fully synthetic. Weekly units are built from a simple, explainable formula:

    mean units = outlet scale x SKU popularity x pack/channel affinity x outlet-SKU taste
                 x seasonality x Ramadan x trend x promo lift

and the observed units are drawn around that mean with Gamma-Poisson (negative binomial) noise.
Stock-outs then overwrite some weeks with zero sales.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import sqlalchemy as sa

# --------------------------------------------------------------------------------------
# Constants: the "world" we simulate
# --------------------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "sales.db"
TRUE_DEMAND_FILE = "true_demand.csv.gz"  # next to the db; git-ignored, rebuilt from the seed

N_WEEKS = 104  # two years of weekly data
WEEKS = pd.date_range("2024-01-01", periods=N_WEEKS, freq="W-MON")  # week_start (Mondays)
START = WEEKS[0]

# Ramadan (first day, last day). Real dates, so the Ramadan flag in Phase 2 is meaningful.
RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-29")]

# Egypt raised list prices a few times; we model two steps (date, relative increase).
PRICE_INCREASES = [("2024-07-01", 0.10), ("2025-04-07", 0.08)]

STOCKOUT_RATE = 0.015  # chance a stocked SKU starts a stock-out in a given week
NOISE_SHAPE = 8.0  # Gamma shape of demand noise: higher = less noisy
PROMO_DISCOUNTS = [10, 15, 20, 25]  # % off, as allowed by the promotions policy
PROMO_DISCOUNT_PROBS = [0.30, 0.30, 0.25, 0.15]
MIN_WEEKS_BETWEEN_PROMOS = 4  # per SKU, as required by the promotions policy


class Region(NamedTuple):
    share: float  # share of outlets
    volume: float  # demand multiplier (Cairo is bigger than the Delta, etc.)
    summer_amp: float  # strength of the summer peak (Upper Egypt is hotter)


class Channel(NamedTuple):
    n_outlets: int
    base_units: float  # weekly units of a popularity-1 SKU in an average outlet
    sku_range: tuple[int, int]  # how many SKUs an outlet of this channel stocks
    ramadan_lift: float  # extra demand at the peak of Ramadan


REGIONS = {
    "Cairo": Region(0.30, 1.15, 0.30),
    "Giza": Region(0.20, 1.05, 0.30),
    "Alexandria": Region(0.15, 1.00, 0.35),
    "Delta": Region(0.20, 0.90, 0.30),
    "Upper Egypt": Region(0.15, 0.80, 0.40),
}

CHANNELS = {
    "kiosk": Channel(130, 14, (6, 10), 0.10),
    "supermarket": Channel(80, 35, (14, 20), 0.20),
    "restaurant_cafe": Channel(60, 20, (6, 10), 0.10),
    "hypermarket": Channel(30, 100, (22, 25), 0.30),
}

# How much each channel favours each pack type (1.0 = neutral).
PACK_AFFINITY = {
    "kiosk": {"can": 0.6, "small": 1.6, "medium": 1.0, "large": 0.5},
    "supermarket": {"can": 1.0, "small": 1.0, "medium": 1.1, "large": 1.2},
    "restaurant_cafe": {"can": 1.6, "small": 1.2, "medium": 0.6, "large": 0.2},
    "hypermarket": {"can": 1.0, "small": 0.7, "medium": 1.0, "large": 1.8},
}

# brand, flavour, category, pack_size, pack_type, launch price (EGP), popularity
SKU_CATALOG = [
    ("Coca-Cola", "Original", "Cola", "330ml can", "can", 20, 3.0),
    ("Coca-Cola", "Original", "Cola", "500ml PET", "small", 17, 2.5),
    ("Coca-Cola", "Original", "Cola", "250ml glass", "small", 14, 0.8),
    ("Coca-Cola", "Original", "Cola", "1L PET", "medium", 28, 2.2),
    ("Coca-Cola", "Original", "Cola", "2.25L PET", "large", 48, 1.8),
    ("Coca-Cola", "Zero", "Cola", "330ml can", "can", 20, 1.4),
    ("Coca-Cola", "Zero", "Cola", "1L PET", "medium", 28, 0.9),
    ("Coca-Cola", "Zero", "Cola", "2.25L PET", "large", 48, 0.7),
    ("Sprite", "Lemon-Lime", "Lemon-lime", "330ml can", "can", 20, 1.6),
    ("Sprite", "Lemon-Lime", "Lemon-lime", "1L PET", "medium", 28, 1.2),
    ("Sprite", "Lemon-Lime", "Lemon-lime", "2.25L PET", "large", 48, 1.0),
    ("Sprite", "Zero", "Lemon-lime", "330ml can", "can", 20, 0.5),
    ("Fanta", "Orange", "Orange", "330ml can", "can", 20, 1.5),
    ("Fanta", "Orange", "Orange", "1L PET", "medium", 28, 1.1),
    ("Fanta", "Orange", "Orange", "2.25L PET", "large", 48, 0.9),
    ("Fanta", "Strawberry", "Fruit", "330ml can", "can", 20, 0.8),
    ("Fanta", "Strawberry", "Fruit", "1L PET", "medium", 28, 0.6),
    ("Fanta", "Pineapple", "Fruit", "1L PET", "medium", 28, 0.5),
    ("Schweppes", "Pomegranate", "Mixer", "330ml can", "can", 22, 0.9),
    ("Schweppes", "Pomegranate", "Mixer", "1L PET", "medium", 30, 0.6),
    ("Schweppes", "Ginger Ale", "Mixer", "330ml can", "can", 22, 0.4),
    ("Powerade", "Mountain Blast", "Sports drink", "500ml PET", "small", 22, 0.8),
    ("Powerade", "Orange", "Sports drink", "500ml PET", "small", 22, 0.5),
    ("Dasani", "Still Water", "Water", "600ml PET", "small", 8, 2.0),
    ("Dasani", "Still Water", "Water", "1.5L PET", "medium", 12, 1.5),
]

# --------------------------------------------------------------------------------------
# Database schema (SQLAlchemy Core). Swap the URL to Postgres later; this stays the same.
# --------------------------------------------------------------------------------------

metadata = sa.MetaData()

outlets_table = sa.Table(
    "outlets",
    metadata,
    sa.Column("outlet_id", sa.String, primary_key=True),
    sa.Column("outlet_name", sa.String, nullable=False),
    sa.Column("channel", sa.String, nullable=False),
    sa.Column("region", sa.String, nullable=False),
    sa.Column("opened_date", sa.Date, nullable=False),
)

skus_table = sa.Table(
    "skus",
    metadata,
    sa.Column("sku_id", sa.String, primary_key=True),
    sa.Column("brand", sa.String, nullable=False),
    sa.Column("flavour", sa.String, nullable=False),
    sa.Column("category", sa.String, nullable=False),
    sa.Column("pack_size", sa.String, nullable=False),
    sa.Column("pack_type", sa.String, nullable=False),
    sa.Column("price_egp", sa.Float, nullable=False),
)

promotions_table = sa.Table(
    "promotions",
    metadata,
    sa.Column("promo_id", sa.Integer, primary_key=True),
    sa.Column("sku_id", sa.String, sa.ForeignKey("skus.sku_id"), nullable=False),
    sa.Column("channel", sa.String),  # NULL = runs in all channels
    sa.Column("start_week", sa.Date, nullable=False),  # week_start of the first promo week
    sa.Column("end_week", sa.Date, nullable=False),  # week_start of the last promo week
    sa.Column("discount_pct", sa.Integer, nullable=False),
)

sales_weekly_table = sa.Table(
    "sales_weekly",
    metadata,
    sa.Column("week_start", sa.Date, nullable=False),
    sa.Column("outlet_id", sa.String, sa.ForeignKey("outlets.outlet_id"), nullable=False),
    sa.Column("sku_id", sa.String, sa.ForeignKey("skus.sku_id"), nullable=False),
    sa.Column("units", sa.Integer, nullable=False),
    sa.Column("unit_price_egp", sa.Float, nullable=False),  # price actually charged that week
    sa.Column("promo_id", sa.Integer, sa.ForeignKey("promotions.promo_id")),
    sa.Column("revenue_egp", sa.Float, nullable=False),
    sa.PrimaryKeyConstraint("week_start", "outlet_id", "sku_id"),
    sa.Index("ix_sales_outlet_sku", "outlet_id", "sku_id"),
)

# --------------------------------------------------------------------------------------
# Dimension tables
# --------------------------------------------------------------------------------------


def make_outlets(rng: np.random.Generator) -> pd.DataFrame:
    """Create ~300 outlets with a channel, a region and an opening date."""
    region_names = list(REGIONS)
    region_probs = [r.share for r in REGIONS.values()]

    rows = []
    for channel, profile in CHANNELS.items():
        regions = rng.choice(region_names, size=profile.n_outlets, p=region_probs)
        rows += [{"channel": channel, "region": region} for region in regions]
    outlets = pd.DataFrame(rows)

    # Shuffle so outlet ids are not sorted by channel.
    outlets = outlets.iloc[rng.permutation(len(outlets))].reset_index(drop=True)
    outlets["outlet_id"] = [f"O{i + 1:03d}" for i in range(len(outlets))]

    label = {"kiosk": "Kiosk", "supermarket": "Market", "restaurant_cafe": "Cafe"}
    label["hypermarket"] = "Hyper"
    outlets["outlet_name"] = [
        f"{r.region} {label[r.channel]} {r.outlet_id[1:]}" for r in outlets.itertuples()
    ]

    # ~8% of outlets open during the 2 years (cold-start cases); the rest are long-standing.
    is_new = rng.random(len(outlets)) < 0.08
    new_week = rng.integers(8, 90, size=len(outlets))
    old_days = rng.integers(30, 3000, size=len(outlets))
    outlets["opened_date"] = np.where(
        is_new,
        WEEKS[new_week],
        START - pd.to_timedelta(old_days, unit="D"),
    )
    outlets["opened_date"] = pd.to_datetime(outlets["opened_date"])
    return outlets[["outlet_id", "outlet_name", "channel", "region", "opened_date"]]


def make_skus() -> pd.DataFrame:
    """Create the 25-SKU catalogue. `popularity` is a generator-only column (not stored)."""
    columns = ["brand", "flavour", "category", "pack_size", "pack_type", "price_egp", "popularity"]
    skus = pd.DataFrame(SKU_CATALOG, columns=columns)
    skus.insert(0, "sku_id", [f"S{i + 1:02d}" for i in range(len(skus))])
    return skus


def make_promotions(rng: np.random.Generator, skus: pd.DataFrame) -> pd.DataFrame:
    """Create 3-6 promotions per SKU, 1-3 weeks long, never closer than 4 weeks apart."""
    channel_names = list(CHANNELS)
    rows = []
    for sku_id in skus["sku_id"]:
        week = int(rng.integers(0, 15))  # first possible start week
        for _ in range(int(rng.integers(3, 7))):
            length = int(rng.integers(1, 4))
            if week + length > N_WEEKS:
                break
            # Half the promos run in every channel, half in a single channel.
            channel = None if rng.random() < 0.5 else str(rng.choice(channel_names))
            rows.append(
                {
                    "sku_id": sku_id,
                    "channel": channel,
                    "start_week": WEEKS[week],
                    "end_week": WEEKS[week + length - 1],
                    "discount_pct": int(rng.choice(PROMO_DISCOUNTS, p=PROMO_DISCOUNT_PROBS)),
                }
            )
            week += length + MIN_WEEKS_BETWEEN_PROMOS + int(rng.integers(0, 16))
    promos = pd.DataFrame(rows)
    promos.insert(0, "promo_id", range(1, len(promos) + 1))
    return promos


# --------------------------------------------------------------------------------------
# Calendar effects (deterministic, one value per week)
# --------------------------------------------------------------------------------------


def ramadan_share() -> np.ndarray:
    """Share of each week (0..1) that falls inside Ramadan."""
    share = np.zeros(N_WEEKS)
    for first_day, last_day in RAMADAN:
        for day in pd.date_range(first_day, last_day):
            week = (day - START).days // 7
            if 0 <= week < N_WEEKS:
                share[week] += 1 / 7
    return share


def price_index() -> np.ndarray:
    """Multiplier on launch list prices for each week (1.0 before the first increase)."""
    index = np.ones(N_WEEKS)
    for date, increase in PRICE_INCREASES:
        index[WEEKS >= pd.Timestamp(date)] *= 1 + increase
    return index


def stockout_mask(rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    """Boolean (pairs x weeks): True where the outlet had no stock of that SKU."""
    starts = rng.random((n_pairs, N_WEEKS)) < STOCKOUT_RATE
    duration = rng.integers(1, 4, size=(n_pairs, N_WEEKS))  # 1-3 weeks
    mask = np.zeros_like(starts)
    for k in range(3):  # a stock-out that started k weeks ago and lasts > k weeks is still on
        still_out = starts & (duration > k)
        mask[:, k:] |= still_out[:, : N_WEEKS - k]
    return mask


# --------------------------------------------------------------------------------------
# Fact table
# --------------------------------------------------------------------------------------


def choose_assortments(
    rng: np.random.Generator, outlets: pd.DataFrame, skus: pd.DataFrame
) -> pd.DataFrame:
    """Decide which SKUs each outlet stocks. Returns one row per (outlet, SKU) pair."""
    pairs = []
    for outlet_idx, channel in enumerate(outlets["channel"]):
        low, high = CHANNELS[channel].sku_range
        n_skus = int(rng.integers(low, high + 1))
        affinity = skus["pack_type"].map(PACK_AFFINITY[channel]).to_numpy()
        weights = skus["popularity"].to_numpy() * affinity
        chosen = rng.choice(len(skus), size=n_skus, replace=False, p=weights / weights.sum())
        pairs += [(outlet_idx, sku_idx) for sku_idx in sorted(chosen)]
    return pd.DataFrame(pairs, columns=["outlet_idx", "sku_idx"])


def promo_arrays(promos: pd.DataFrame, skus: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Discount fraction and promo_id per (SKU, channel, week); 0 means no promotion."""
    shape = (len(skus), len(CHANNELS), N_WEEKS)
    discount, promo_id = np.zeros(shape), np.zeros(shape, dtype=int)
    sku_pos = {sku_id: i for i, sku_id in enumerate(skus["sku_id"])}
    channel_pos = {name: i for i, name in enumerate(CHANNELS)}

    for p in promos.itertuples():
        first = (p.start_week - START).days // 7
        last = (p.end_week - START).days // 7
        channels = list(channel_pos.values()) if pd.isna(p.channel) else [channel_pos[p.channel]]
        for c in channels:
            discount[sku_pos[p.sku_id], c, first : last + 1] = p.discount_pct / 100
            promo_id[sku_pos[p.sku_id], c, first : last + 1] = p.promo_id
    return discount, promo_id


def build_sales(
    rng: np.random.Generator, outlets: pd.DataFrame, skus: pd.DataFrame, promos: pd.DataFrame
) -> pd.DataFrame:
    """Simulate weekly units for every stocked outlet x SKU pair."""
    pairs = choose_assortments(rng, outlets, skus)
    o, s = pairs["outlet_idx"].to_numpy(), pairs["sku_idx"].to_numpy()
    n_pairs = len(pairs)

    channel = outlets["channel"].to_numpy()[o]
    region = outlets["region"].to_numpy()[o]
    channel_idx = np.array([list(CHANNELS).index(c) for c in channel])
    region_idx = np.array([list(REGIONS).index(r) for r in region])

    # 1) Static scale of each pair: bigger outlets sell more, popular SKUs sell more.
    outlet_scale = rng.lognormal(0, 0.4, size=len(outlets))
    taste = rng.lognormal(0, 0.35, size=n_pairs)  # stable outlet x SKU preference
    base_units = np.array([CHANNELS[c].base_units for c in channel])
    region_volume = np.array([REGIONS[r].volume for r in region])
    affinity = np.array(
        [PACK_AFFINITY[c][p] for c, p in zip(channel, skus["pack_type"].to_numpy()[s], strict=True)]
    )
    scale = (
        base_units * region_volume * outlet_scale[o] * skus["popularity"].to_numpy()[s]
        * affinity * taste
    )  # fmt: skip

    # 2) Calendar effects, shape (pairs, weeks).
    t = np.arange(N_WEEKS)
    week_of_year = WEEKS.isocalendar().week.to_numpy().astype(float)
    summer_amp = np.array([r.summer_amp for r in REGIONS.values()])
    season = 1 + summer_amp[region_idx, None] * np.sin(2 * np.pi * (week_of_year - 15) / 52)
    ramadan_lift = np.array([c.ramadan_lift for c in CHANNELS.values()])
    ramadan = 1 + ramadan_lift[channel_idx, None] * ramadan_share()
    growth = rng.normal(0.06, 0.05, size=len(outlets))  # yearly growth per outlet
    trend = (1 + growth[o, None]) ** (t / 52)

    # 3) Promotions: a discount raises volume by (1 - d) ** -elasticity.
    discount_arr, promo_id_arr = promo_arrays(promos, skus)
    discount = discount_arr[s, channel_idx]  # (pairs, weeks)
    promo_id = promo_id_arr[s, channel_idx]
    elasticity = rng.uniform(1.5, 2.5, size=len(skus))
    promo_lift = (1 - discount) ** -elasticity[s, None]

    # 4) Mean demand, then noise around it (Gamma-Poisson = negative binomial).
    mean_units = scale[:, None] * season * ramadan * trend * promo_lift
    noise = rng.gamma(NOISE_SHAPE, 1 / NOISE_SHAPE, size=mean_units.shape)
    units = rng.poisson(mean_units * noise)
    units[stockout_mask(rng, n_pairs)] = 0

    # 5) Keep only weeks after the outlet opened, and flatten to one row per outlet-SKU-week.
    open_week = np.ceil((outlets["opened_date"] - START).dt.days.to_numpy() / 7).clip(min=0)
    pair_idx, week_idx = np.nonzero(t[None, :] >= open_week[o, None])

    list_price = skus["price_egp"].to_numpy()[s, None] * price_index()
    unit_price = np.round(list_price * (1 - discount), 2)
    row_units = units[pair_idx, week_idx]
    row_price = unit_price[pair_idx, week_idx]
    row_promo = promo_id[pair_idx, week_idx]
    return pd.DataFrame(
        {
            "week_start": WEEKS[week_idx],
            "outlet_id": outlets["outlet_id"].to_numpy()[o[pair_idx]],
            "sku_id": skus["sku_id"].to_numpy()[s[pair_idx]],
            "units": row_units,
            "unit_price_egp": row_price,
            "promo_id": pd.array(np.where(row_promo > 0, row_promo, pd.NA), dtype="Int64"),
            "revenue_egp": np.round(row_units * row_price, 2),
            # The true mean demand BEFORE noise and stock-outs. Not part of the database: it is
            # saved separately (main) and only used to measure the forecast "oracle" floor.
            "expected_units": np.round(mean_units[pair_idx, week_idx], 4),
        }
    )


# --------------------------------------------------------------------------------------
# Load into SQLite and report
# --------------------------------------------------------------------------------------


def insert_df(conn: sa.Connection, table: sa.Table, df: pd.DataFrame, chunk: int = 50_000) -> None:
    """Insert a DataFrame into `table` in chunks, using plain Python types (dates, None)."""
    df = df[[c.name for c in table.columns]].copy()
    for col in df.select_dtypes("datetime").columns:
        df[col] = df[col].dt.date
    for start in range(0, len(df), chunk):
        part = df.iloc[start : start + chunk].astype(object)
        part = part.where(part.notna(), None)
        conn.execute(table.insert(), part.to_dict("records"))


def write_db(db_path: Path, **tables: pd.DataFrame) -> sa.Engine:
    """(Re)create the SQLite database and load all four tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    engine = sa.create_engine(f"sqlite:///{db_path}")
    metadata.create_all(engine)
    with engine.begin() as conn:
        # Parents first so foreign keys always point at existing rows.
        insert_df(conn, outlets_table, tables["outlets"])
        insert_df(conn, skus_table, tables["skus"])
        insert_df(conn, promotions_table, tables["promotions"])
        insert_df(conn, sales_weekly_table, tables["sales_weekly"])
    return engine


def print_summary(engine: sa.Engine) -> None:
    """Print row counts and a few sanity checks that the simulated effects are visible."""
    queries = {
        "Row counts": """
            SELECT 'outlets' AS tbl, COUNT(*) AS n FROM outlets
            UNION ALL SELECT 'skus', COUNT(*) FROM skus
            UNION ALL SELECT 'promotions', COUNT(*) FROM promotions
            UNION ALL SELECT 'sales_weekly', COUNT(*) FROM sales_weekly""",
        "Outlets by channel": "SELECT channel, COUNT(*) AS n FROM outlets GROUP BY channel",
        "Outlets by region": "SELECT region, COUNT(*) AS n FROM outlets GROUP BY region",
        "Units by month (seasonality + Ramadan in Mar-Apr)": """
            SELECT substr(week_start, 1, 7) AS month, SUM(units) AS units
            FROM sales_weekly GROUP BY month ORDER BY month""",
        "Avg units per row: promo vs normal weeks": """
            SELECT promo_id IS NOT NULL AS is_promo_week,
                   ROUND(AVG(units), 2) AS avg_units, COUNT(*) AS n
            FROM sales_weekly WHERE units > 0 GROUP BY is_promo_week""",
        "Share of zero-sales rows (stock-outs)": """
            SELECT ROUND(100.0 * SUM(units = 0) / COUNT(*), 2) AS pct_zero FROM sales_weekly""",
    }
    with engine.connect() as conn:
        for title, sql in queries.items():
            print(f"\n== {title}")
            print(pd.read_sql(sa.text(sql), conn).to_string(index=False))


def write_samples(out_dir: Path, seed: int, n_rows: int = 200, **tables: pd.DataFrame) -> None:
    """Write a small CSV preview of each table (committed to git, so the repo is browsable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        sample = df if len(df) <= n_rows else df.sample(n_rows, random_state=seed).sort_index()
        sample.drop(columns=["popularity"], errors="ignore").to_csv(
            out_dir / f"{name}.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_DB.parent / "sample")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outlets = make_outlets(rng)
    skus = make_skus()
    promos = make_promotions(rng, skus)
    sales = build_sales(rng, outlets, skus, promos)

    # The simulator's hidden truth lives in its own file, so it can never leak into the
    # database that the SQL tool (and therefore the LLM) can query.
    true_path = args.db.parent / TRUE_DEMAND_FILE
    key = ["week_start", "outlet_id", "sku_id"]
    sales[[*key, "expected_units"]].to_csv(true_path, index=False)
    sales = sales.drop(columns="expected_units")

    engine = write_db(args.db, outlets=outlets, skus=skus, promotions=promos, sales_weekly=sales)
    write_samples(
        args.sample_dir,
        args.seed,
        outlets=outlets,
        skus=skus,
        promotions=promos,
        sales_weekly=sales,
    )
    print(f"Wrote {args.db} and CSV samples in {args.sample_dir} (seed={args.seed})")
    print_summary(engine)


if __name__ == "__main__":
    main()
