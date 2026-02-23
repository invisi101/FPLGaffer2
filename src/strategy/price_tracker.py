"""Price tracking and prediction.

Ported from v1 season_manager.py (track_prices, get_price_alerts,
predict_price_changes, get_price_history).
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Thresholds (from v1, kept as module-level constants) ──────────────
_MIN_TOTAL_TRANSFERS = 5_000
_NET_RISE_THRESHOLD = 20_000
_NET_FALL_THRESHOLD = -20_000
_MIN_OWNERSHIP_PCT = 0.1
_RATIO_THRESHOLD = 0.005


def track_prices(
    season_id: int,
    bootstrap: dict,
    squad_ids: set[int],
    watchlist_ids: set[int] | None = None,
) -> list[dict]:
    """Build price snapshot records for squad + watchlist players.

    Returns a list of dicts ready for bulk insertion into price_tracker.
    Does NOT write to DB -- the caller is responsible for persisting.
    """
    elements = bootstrap.get("elements", [])
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}

    track_ids = set(squad_ids)
    if watchlist_ids:
        track_ids |= watchlist_ids

    players: list[dict] = []
    for el in elements:
        if el["id"] in track_ids:
            tid = el.get("team")
            players.append({
                "season_id": season_id,
                "player_id": el["id"],
                "web_name": el.get("web_name"),
                "team_code": id_to_code.get(tid),
                "price": el.get("now_cost", 0) / 10,
                "transfers_in_event": el.get("transfers_in_event", 0),
                "transfers_out_event": el.get("transfers_out_event", 0),
            })

    return players


def get_price_alerts(bootstrap: dict) -> list[dict]:
    """Flag players likely to rise/fall based on transfer volume.

    Returns sorted list of {player_id, web_name, team, price,
    net_transfers, direction}.
    """
    elements = bootstrap.get("elements", [])
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    code_to_short = {
        t["code"]: t["short_name"] for t in bootstrap.get("teams", [])
    }

    alerts: list[dict] = []
    for el in elements:
        net = el.get("transfers_in_event", 0) - el.get(
            "transfers_out_event", 0,
        )
        total_transfers = el.get("transfers_in_event", 0) + el.get(
            "transfers_out_event", 0,
        )

        if total_transfers < _MIN_TOTAL_TRANSFERS:
            continue

        if net > _NET_RISE_THRESHOLD:
            tc = id_to_code.get(el.get("team"))
            alerts.append({
                "player_id": el["id"],
                "web_name": el.get("web_name"),
                "team": code_to_short.get(tc, ""),
                "price": el.get("now_cost", 0) / 10,
                "net_transfers": net,
                "direction": "rise",
            })
        elif net < _NET_FALL_THRESHOLD:
            tc = id_to_code.get(el.get("team"))
            alerts.append({
                "player_id": el["id"],
                "web_name": el.get("web_name"),
                "team": code_to_short.get(tc, ""),
                "price": el.get("now_cost", 0) / 10,
                "net_transfers": net,
                "direction": "fall",
            })

    alerts.sort(key=lambda a: abs(a["net_transfers"]), reverse=True)
    return alerts


def predict_price_changes(bootstrap: dict) -> list[dict]:
    """Predict price changes using ownership-based algorithm approximation.

    Uses: transfer_ratio = net_transfers / (ownership_pct * 100_000)
    Rise if ratio > 0.005, fall if < -0.005.
    Probability = percentile rank of abs(ratio) among all qualifying players.
    """
    elements = bootstrap.get("elements", [])
    id_to_code = {t["id"]: t["code"] for t in bootstrap.get("teams", [])}
    code_to_short = {
        t["code"]: t["short_name"] for t in bootstrap.get("teams", [])
    }

    raw: list[dict] = []
    for el in elements:
        net = el.get("transfers_in_event", 0) - el.get(
            "transfers_out_event", 0,
        )
        ownership = el.get("selected_by_percent")
        if ownership is None:
            continue
        try:
            ownership_pct = float(ownership)
        except (TypeError, ValueError):
            continue

        if ownership_pct < _MIN_OWNERSHIP_PCT:
            continue

        transfer_ratio = net / (ownership_pct * 100_000)

        if abs(transfer_ratio) < _RATIO_THRESHOLD:
            continue

        direction = "rise" if transfer_ratio > 0 else "fall"
        estimated_change = 0.1 if direction == "rise" else -0.1

        tc = id_to_code.get(el.get("team"))
        raw.append({
            "player_id": el["id"],
            "web_name": el.get("web_name", "Unknown"),
            "team": code_to_short.get(tc, ""),
            "price": el.get("now_cost", 0) / 10,
            "ownership": ownership_pct,
            "net_transfers": net,
            "transfer_ratio": round(transfer_ratio, 6),
            "abs_ratio": abs(transfer_ratio),
            "direction": direction,
            "estimated_change": estimated_change,
        })

    # Probability = percentile rank among qualifying players
    if raw:
        sorted_ratios = sorted(r["abs_ratio"] for r in raw)
        n = len(sorted_ratios)
        for r in raw:
            # bisect to find position, convert to 0.5-1.0 range
            rank = bisect.bisect_left(sorted_ratios, r["abs_ratio"])
            r["probability"] = (
                round(0.5 + 0.5 * rank / (n - 1), 3) if n > 1 else 1.0
            )
            del r["abs_ratio"]

    predictions = raw
    predictions.sort(key=lambda p: p["probability"], reverse=True)
    return predictions


def get_price_history(
    all_history: list[dict],
    player_ids: list[int] | None = None,
    days: int = 14,
) -> dict:
    """Build price history from raw price_tracker rows.

    Parameters
    ----------
    all_history:
        Raw rows from the price_tracker table (list of dicts with
        player_id, web_name, snapshot_date, price, transfers_in_event,
        transfers_out_event).
    player_ids:
        Optional filter -- only include these player IDs.
    days:
        Number of days to look back.

    Returns
    -------
    {player_id: {web_name, snapshots: [{date, price, net_transfers}]}}.
    """
    if not all_history:
        return {}

    # Filter by player_ids if provided
    if player_ids:
        pid_set = set(player_ids)
        all_history = [h for h in all_history if h["player_id"] in pid_set]

    # Filter by days
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    all_history = [
        h for h in all_history if h.get("snapshot_date", "") >= cutoff
    ]

    # Group by player_id
    result: dict = {}
    for h in all_history:
        pid = h["player_id"]
        if pid not in result:
            result[pid] = {
                "web_name": h.get("web_name", "Unknown"),
                "snapshots": [],
            }
        net = (h.get("transfers_in_event") or 0) - (
            h.get("transfers_out_event") or 0
        )
        result[pid]["snapshots"].append({
            "date": h.get("snapshot_date"),
            "price": h.get("price"),
            "net_transfers": net,
        })

    return result
