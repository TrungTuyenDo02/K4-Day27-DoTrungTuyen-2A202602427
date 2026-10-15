"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    >= 73%       auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


async def fetch_recent_threads(limit: int = 10) -> list[dict]:
    """Same query as audit.replay.list_threads, but returns rows instead of printing."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MAX(timestamp)  AS last_event,
                   MAX(risk_level) AS worst_risk,
                   COUNT(*)        AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


# ─── Session state ─────────────────────────────────────────────────────────
for _key, _default in [
    ("thread_id", None), ("pr_url", ""), ("interrupt_payload", None), ("final", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str):
    """Rebuild a session from the checkpointer.

    Returns (pending interrupt payload | None, final state values | None, pr_url).
    """
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    values = snap.values or {}
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value, None, values.get("pr_url", "")
    if not snap.next and values:
        return None, values, values.get("pr_url", "")
    return None, None, values.get("pr_url", "")


def open_thread(thread_id: str, pr_url: str = "") -> None:
    """Point the UI at `thread_id` (state + URL) and restore it from the checkpointer."""
    payload, final, saved_url = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = saved_url or pr_url
    st.session_state.interrupt_payload = payload
    st.session_state.final = final
    st.query_params["thread"] = thread_id


# A browser reload wipes session_state: rebuild it from ?thread=<id>.
_url_thread = st.query_params.get("thread")
if _url_thread and st.session_state.thread_id is None:
    open_thread(_url_thread)


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    recent = asyncio.run(fetch_recent_threads())
    if not recent:
        st.caption("No sessions yet.")
    else:
        st.dataframe(
            [
                {
                    "thread": r["thread_id"][:8],
                    "PR": "#" + r["pr_url"].rsplit("/", 1)[-1],
                    "risk": r["worst_risk"],
                    "events": r["events"],
                    "last event": r["last_event"],
                }
                for r in recent
            ],
            hide_index=True,
            width="stretch",
        )
        by_label = {
            f"PR #{r['pr_url'].rsplit('/', 1)[-1]} · {r['worst_risk']} · {r['thread_id'][:8]}": r
            for r in recent
        }
        picked = st.selectbox("Open a session", list(by_label), key="session_picker")
        if st.button("Open", key="open_session"):
            r = by_label[picked]
            open_thread(r["thread_id"], r["pr_url"])
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", key="approve_btn"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", key="reject_btn"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", key="edit_btn"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    st.info(
        "After you submit, the agent refines its review with your answers and posts it "
        "to the PR (answers are included in the public comment)."
    )

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_input(q, key=f"escalation_q_{i}")
        submitted_answers = st.form_submit_button("Submit answers")

    if submitted_answers:
        if any(a.strip() for a in answers.values()):
            return answers
        st.warning("Answer at least one question before submitting.")
    return None


def _field(obj, name: str, default=None):
    """Read a field from a pydantic model or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def render_refined_review(analysis) -> None:
    st.markdown("#### Refined review (posted after escalation)")
    st.caption(f"confidence {_field(analysis, 'confidence', 0):.0%} — "
               f"{_field(analysis, 'confidence_reasoning', '')}")
    st.markdown(_field(analysis, "summary", ""))
    for c in _field(analysis, "comments", []) or []:
        st.markdown(
            f"- **[{_field(c, 'severity')}]** `{_field(c, 'file')}:{_field(c, 'line') or '?'}` "
            f"— {_field(c, 'body')}"
        )


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.query_params["thread"] = st.session_state.thread_id

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action") or "?"
    analysis = final.get("analysis")
    pr_link = st.session_state.pr_url or final.get("pr_url", "")

    if "failed" in action:
        st.error(f"✗ {action} — the review comment could not be posted to {pr_link}")
    elif action.startswith("auto_"):
        st.success(f"✓ Auto-approved — comment posted to {pr_link}")
        if analysis is not None:
            st.metric("Confidence", f"{_field(analysis, 'confidence', 0):.0%}")
            st.markdown(_field(analysis, "summary", ""))
        st.markdown(f"[Open the PR]({pr_link})")
    elif action in ("committed", "committed_after_escalation"):
        st.success(f"✓ {action} — comment posted to {pr_link}")
        st.markdown(f"[Open the PR]({pr_link})")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")

    # The escalation path posts the refined review without a second confirmation.
    if final.get("escalation_answers") and analysis is not None:
        render_refined_review(analysis)

if st.session_state.thread_id:
    st.caption(
        f"thread_id = {st.session_state.thread_id}  ·  replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
