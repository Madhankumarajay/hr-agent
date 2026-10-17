# HR Assistant — LangGraph + FastAPI + React

One chat that does two things:
1. **RAG** – answers handbook/benefits questions from the public PDFs, with **page citations**.
2. **Booking** – multi-turn booking (name → email → slot) using tools.

If you ask a handbook question **in the middle of a booking**, it answers, then **resumes the booking** and remembers what you already gave.

## Architecture

```
React (thread_id, messages, status badge)
   │  POST /api/chat
FastAPI  ──►  LangGraph StateGraph  (MemorySaver / SqliteSaver, keyed by thread_id)

   START ─► router ─┬─► faq (RAG) ──(booking active?)──► booking ─► respond ─► END
                    ├─► booking ───────────────────────────▲
                    ├─► chat ──────────────────────────────────────► respond
                    └─► (blocked by guardrail) ────────────────────► respond
```

| File | What |
|---|---|
| `backend/app/graph.py` | State schema, nodes, router, edges, checkpointer |
| `backend/app/rag.py` | PDF → pypdf → chunks (800/150) → ChromaDB → search |
| `backend/app/tools.py` | `list_available_slots`, `create_booking` tools |
| `backend/app/main.py` | `POST /api/chat`, `GET /api/slots`, `GET /api/health` |
| `frontend/src/App.jsx` | Chat UI |

**State** (`State` TypedDict): `messages` + `booking` persist per `thread_id`; `route`, `parts`, `citations`, `trace` are per-turn scratch values reset every turn.

**How interleaving works** – the router (one LLM call, structured output) can flag a *question* **and** booking details in the same message. Graph runs `faq` first, then `booking`, which continues from the saved `booking` state. The booking logic itself is plain code (the LLM only *understands* messages), so bookings are reliable. Names must appear in the user's text (anti-hallucination), emails come from regex.

**Guardrail** – regex check in `router` refuses prompt-injection / "show your system prompt" attempts and leaves booking state untouched.

## Run

```bash
# 1. key (free): https://console.groq.com/keys
cp .env.example .env        # put GROQ_API_KEY in it

# 2. backend (http://localhost:8000)
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txtuvicorn app.main:app --port 8000

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
        
#  first start downloads the PDFs + embedding model and indexes them (~1 min)

# 3. frontend (http://localhost:5173)
cd frontend
npm install
npm run dev
```


Tests (no key, no network): `cd backend && pytest -q`
No `GROQ_API_KEY`? It still runs with rule-based routing and extractive answers.

## Demo script (sample dialogue)
1. `Hi, I need to book a 1-on-1 Performance Review session.`
2. `Alex Chen (alex.chen@example.com). Wait, before that, how often are performance reviews conducted according to the handbook?`
3. `Let's go with Friday at 10:30 AM.`

## AI Engineering Log
> Fill with YOUR real experience (they grade honesty). Delete this note after.

**Tools used:** Claude (scaffold of backend/frontend), _add others_

| # | What went wrong (AI output) | How I found it | Fix |
|---|---|---|---|
| 1 | _e.g. booking state lost after FAQ turn_ | _printed graph state_ | _…_ |
| 2 | | | |
| 3 | | | |

## Verification artifacts
Add screenshots / terminal logs here: RAG answer with citation, multi-turn booking, FAQ-mid-booking resume, injection refused, `pytest` output.
