"""Score six saved meetings, resource reservations and deadline eligibility."""

from copy import deepcopy

from benchmark.office.cases.schedule import resource_case


def check(reference, after):
    # Reuse exact field/resource matching from the four-goal case, then derive
    # denominators and whole-task constraints from this case's target set.
    result = resource_case.check(reference, after)
    total = len(reference["targets"])
    result.update(total=total, completion_rate=result["completed"] / total)
    result["all_backend_requirements_met"] = (
        result["completed"] == total
        and result["new_event_count"] == total
        and result["background_unchanged"]
        and result["resource_background_unchanged"]
        and len(after["resources"]["room_reservations"]) == total
    )
    return result


def check_at_deadline(reference, snapshot, writes, *, started_at, budget_seconds):
    """Score an observed snapshot; reject later inserts even if capture is late.

    Callers capture a read-transaction snapshot at the fixed deadline (or at
    terminal completion if earlier). Later state must not replace that snapshot.
    Each event and its resource reservation are committed atomically by the app.
    """
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    deadline = started_at + budget_seconds
    old_ids = {row["id"] for row in reference["before"]["events"]}
    admitted = {row["event_id"] for row in writes if started_at <= row["created"] <= deadline}
    kept_ids = old_ids | admitted
    at_cutoff = deepcopy(snapshot)
    at_cutoff["events"] = [row for row in snapshot["events"] if row["id"] in kept_ids]
    at_cutoff["resources"]["room_reservations"] = [
        row for row in snapshot["resources"]["room_reservations"] if row["event_id"] in kept_ids
    ]
    result = check(reference, at_cutoff)
    result.update(
        budget_seconds=budget_seconds,
        started_at=started_at,
        deadline=deadline,
        admitted_new_event_ids=sorted(admitted - old_ids),
    )
    return result
