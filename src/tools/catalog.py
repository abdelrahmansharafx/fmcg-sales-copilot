"""Small lookups shared by the tools: SKU names, outlet info, id validation."""

from __future__ import annotations

import sqlalchemy as sa

from src.errors import NotFoundError
from src.forecast.predict import get_engine


def sku_names() -> dict[str, str]:
    """sku_id -> readable name, e.g. 'S01' -> 'Coca-Cola Original 330ml can'."""
    query = "SELECT sku_id, brand || ' ' || flavour || ' ' || pack_size FROM skus"
    with get_engine().connect() as conn:
        return {row[0]: row[1] for row in conn.execute(sa.text(query))}


def brand_names() -> list[str]:
    """The brands in the catalogue, e.g. ['Coca-Cola', 'Dasani', ...]."""
    with get_engine().connect() as conn:
        return [r[0] for r in conn.execute(sa.text("SELECT DISTINCT brand FROM skus ORDER BY 1"))]


def require_sku(sku_id: str) -> str:
    """Return the SKU's name or raise a helpful error."""
    names = sku_names()
    if sku_id not in names:
        raise NotFoundError(
            f"Unknown sku_id '{sku_id}'. Valid ids look like S01..S{len(names):02d}."
        )
    return names[sku_id]


def outlet_channel(outlet_id: str) -> str:
    """Return the outlet's channel or raise a helpful error."""
    with get_engine().connect() as conn:
        row = conn.execute(
            sa.text("SELECT channel FROM outlets WHERE outlet_id = :id"), {"id": outlet_id}
        ).first()
    if row is None:
        raise NotFoundError(f"Unknown outlet_id '{outlet_id}'. Valid ids look like O001..O300.")
    return row[0]
