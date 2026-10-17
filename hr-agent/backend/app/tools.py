import random
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

TZ = ZoneInfo("Asia/Kolkata")
_lock = threading.Lock()
_booked = {}


def _slot(day, hour, minute, today):
    dt = datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)
    tomorrow = day == today + timedelta(days=1)
    name = dt.strftime("%A")
    t = dt.strftime("%I:%M %p").lstrip("0")
    return {
        "id": f"SLOT-{dt:%Y%m%d-%H%M}",
        "label": f"{'Tomorrow' if tomorrow else name} {t}",
        "day_label": "Tomorrow" if tomorrow else name,
        "time_label": t,
        "date": dt.strftime("%a, %d %b"),
        "hour": hour,
        "minute": minute,
        "days": [name.lower()] + (["tomorrow"] if tomorrow else []),
    }


def _next(start, weekday):
    while start.weekday() != weekday:
        start += timedelta(days=1)
    return start


def all_slots():
    today = datetime.now(TZ).date()
    later = today + timedelta(days=2)
    s = [_slot(today + timedelta(days=1), 11, 0, today), _slot(today + timedelta(days=1), 15, 0, today),
         _slot(_next(later, 4), 10, 30, today), _slot(_next(later, 0), 14, 0, today)]
    return sorted(s, key=lambda x: x["id"])


@tool
def list_available_slots() -> list:
    """List all currently available booking slots."""
    with _lock:
        return [
        s for s in all_slots()
        if s["id"] not in _booked
        ]

@tool
def create_booking(slot_id: str, name: str, email: str, service: str) -> dict:
    """Book an available appointment slot for a candidate."""
    with _lock:
        slot = next(
            (s for s in all_slots() if s["id"] == slot_id),
            None
        )

        if slot is None or slot_id in _booked:
            return {
                "ok": False,
                "message": "That slot is no longer available."
            }

        ref = f"REV-{random.randint(1000, 9999)}"

        _booked[slot_id] = {
            "ref": ref,
            "name": name,
            "email": email,
            "service": service
        }

        return {
            "ok": True,
            "reference_id": ref,
            "day_label": slot["day_label"],
            "time_label": slot["time_label"],
            "slot_label": slot["label"]
        }

def reset():  # tests
    _booked.clear()
