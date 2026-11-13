"""Score six work-log increments and preservation of existing records."""

from benchmark.office.datasets.timesheet.data import goals


def check(phase, before, after):
    old = {row["id"]: row for row in before["entries"]}
    added = [row for row in after["entries"] if row["id"] not in old]
    items = []
    for target in goals(phase):
        matches = [row for row in added if all(row.get(k) == v for k, v in target.items())]
        delta = sum(r["minutes"] for r in after["entries"] if r["card"] == target["card"]) - sum(
            r["minutes"] for r in before["entries"] if r["card"] == target["card"]
        )
        items.append(
            {
                "card": target["card"],
                "saved": len(matches) == 1,
                "delta_correct": delta == target["minutes"],
            }
        )
    completed = sum(i["saved"] and i["delta_correct"] for i in items)
    unchanged = all(row in after["entries"] for row in old.values())
    return {
        "items": items,
        "completed": completed,
        "new_entries": len(added),
        "background_unchanged": unchanged,
        "all_backend_requirements_met": completed == 6 and len(added) == 6 and unchanged,
    }
