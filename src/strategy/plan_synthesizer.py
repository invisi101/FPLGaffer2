"""Plan synthesis -- combine transfer plan, captain plan, and chip schedule.

Ported from v1 strategy.py PlanSynthesizer class.
Produces a unified strategic plan with natural-language rationale.
"""

from __future__ import annotations

from datetime import datetime

from src.logging_config import get_logger

logger = get_logger(__name__)


class PlanSynthesizer:
    """Combine transfer plan, captain plan, and chip schedule into one
    coherent plan."""

    def synthesize(
        self,
        transfer_plan: list[dict],
        captain_plan: list[dict],
        chip_heatmap: dict[str, dict[int, float]],
        chip_synergies: list[dict],
        available_chips: set[str],
    ) -> dict:
        """Produce a unified strategic plan with natural-language rationale.

        Returns {
            timeline: [{gw, transfers, captain, chip, confidence, rationale}],
            chip_schedule: {chip: gw},
            chip_synergies: [...top 3...],
            rationale: str,
            generated_at: str,
        }
        """
        # Determine optimal chip schedule from heatmap + synergies
        chip_schedule = self._plan_chip_schedule(
            chip_heatmap, chip_synergies, available_chips,
        )

        # Build unified timeline
        all_gws: set[int] = set()
        for step in transfer_plan:
            all_gws.add(step["gw"])
        for cap in captain_plan:
            all_gws.add(cap["gw"])

        timeline: list[dict] = []
        for gw in sorted(all_gws):
            entry: dict = {"gw": gw}

            # Transfer info
            transfer_step = next(
                (s for s in transfer_plan if s["gw"] == gw), None,
            )
            if transfer_step:
                entry["transfers_in"] = transfer_step.get("transfers_in", [])
                entry["transfers_out"] = transfer_step.get("transfers_out", [])
                entry["ft_used"] = transfer_step.get("ft_used", 0)
                entry["ft_available"] = transfer_step.get("ft_available", 0)
                entry["transfer_rationale"] = transfer_step.get("rationale", "")
                entry["predicted_points"] = transfer_step.get(
                    "predicted_points", 0,
                )
                entry["squad_ids"] = transfer_step.get("squad_ids", [])
                if transfer_step.get("new_squad"):
                    entry["new_squad"] = transfer_step["new_squad"]

            # Captain info
            cap_step = next(
                (c for c in captain_plan if c["gw"] == gw), None,
            )
            if cap_step:
                entry["captain_id"] = cap_step["captain_id"]
                entry["captain_name"] = cap_step["captain_name"]
                entry["captain_points"] = cap_step["captain_points"]
                entry["vc_id"] = cap_step["vc_id"]
                entry["vc_name"] = cap_step["vc_name"]
                entry["weak_captain"] = cap_step.get("weak_gw", False)

            # Chip info
            for chip_name, chip_gw in chip_schedule.items():
                if chip_gw == gw:
                    entry["chip"] = chip_name
                    chip_val = chip_heatmap.get(chip_name, {}).get(gw, 0)
                    entry["chip_value"] = chip_val

            # Confidence (based on distance from current GW)
            entry["confidence"] = (
                cap_step.get("confidence", 0.9) if cap_step else 0.9
            )

            timeline.append(entry)

        # Build overall rationale
        rationale = self._build_rationale(
            timeline, chip_schedule, chip_synergies,
        )

        return {
            "timeline": timeline,
            "chip_schedule": chip_schedule,
            "chip_synergies": chip_synergies[:3],  # Top 3
            "rationale": rationale,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _plan_chip_schedule(
        self,
        chip_heatmap: dict[str, dict[int, float]],
        chip_synergies: list[dict],
        available_chips: set[str],
    ) -> dict[str, int]:
        """Determine which GW to play each chip.

        Uses synergy-aware scheduling: if WC->BB synergy is the top combo,
        schedule both together.
        """
        schedule: dict[str, int] = {}

        # Check if top synergy is worth using
        if chip_synergies:
            top_syn = chip_synergies[0]
            # Use synergy if combined value > sum of individual peak values * 0.9
            syn_chips = top_syn["chips"]
            syn_gws = top_syn["gws"]
            individual_peaks: list[float] = []
            for chip in syn_chips:
                if chip in chip_heatmap and chip in available_chips:
                    vals = chip_heatmap[chip]
                    if vals:
                        individual_peaks.append(max(vals.values()))
                    else:
                        individual_peaks.append(0)

            if len(individual_peaks) == len(syn_chips):
                peak_sum = sum(individual_peaks)
                if top_syn["combined_value"] > peak_sum * 0.9:
                    # Use synergy schedule
                    for chip, gw in zip(syn_chips, syn_gws):
                        if chip in available_chips:
                            schedule[chip] = gw

        # Schedule remaining chips by their peak GW
        for chip in available_chips:
            if chip in schedule:
                continue
            if chip in chip_heatmap:
                vals = chip_heatmap[chip]
                if vals:
                    # Avoid scheduling on same GW as another chip
                    used_gws = set(schedule.values())
                    sorted_gws = sorted(
                        vals.items(), key=lambda x: x[1], reverse=True,
                    )
                    for gw, val in sorted_gws:
                        if gw not in used_gws:
                            schedule[chip] = gw
                            break

        return schedule

    def _build_rationale(
        self,
        timeline: list[dict],
        chip_schedule: dict[str, int],
        chip_synergies: list[dict],
    ) -> str:
        """Build natural-language summary of the strategic plan."""
        parts: list[str] = []

        if chip_schedule:
            chip_labels = {
                "wildcard": "Wildcard",
                "freehit": "Free Hit",
                "bboost": "Bench Boost",
                "3xc": "Triple Captain",
            }
            chip_parts = [
                f"{chip_labels.get(c, c)} in GW{g}"
                for c, g in chip_schedule.items()
            ]
            parts.append(f"Chip plan: {', '.join(chip_parts)}.")

        if chip_synergies:
            top = chip_synergies[0]
            parts.append(
                f"Key synergy: {top['description']} "
                f"(+{top['synergy_bonus']:.1f} pts bonus)."
            )

        # Summarize transfer strategy
        bank_gws = [t for t in timeline if t.get("ft_used", 0) == 0]
        use_gws = [t for t in timeline if t.get("ft_used", 0) > 0]
        if bank_gws and use_gws:
            parts.append(
                f"Transfer strategy: bank in "
                f"GW{','.join(str(t['gw']) for t in bank_gws)}, "
                f"use in GW{','.join(str(t['gw']) for t in use_gws)}."
            )

        # Flag weak captain GWs
        weak = [t for t in timeline if t.get("weak_captain")]
        if weak:
            parts.append(
                f"Weak captain GW(s): "
                f"{','.join(str(t['gw']) for t in weak)} -- "
                "consider transfer to upgrade premium captain option."
            )

        return (
            " ".join(parts)
            if parts
            else "No significant strategic adjustments needed."
        )
