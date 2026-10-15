"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

One AuditEntry per node. human_approval and escalate log twice: an "awaiting" row
before interrupt() and the outcome after resume. A node re-runs from the top on
resume, so the "before" row goes through `audit_once` to avoid duplicates.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, replay_events, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


async def audit_once(state, entry: AuditEntry) -> None:
    """Write `entry` unless this thread already has a row with the same action+decision.

    A node that calls interrupt() re-runs from the top on resume, so its "before
    interrupt" row would otherwise be written twice.
    """
    for ev in await replay_events(state["thread_id"]):
        if ev["action"] == entry.action and ev["decision"] == entry.decision:
            return
    await audit(state, entry)


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _reviewer() -> str | None:
    return os.environ.get("GITHUB_USER")


ANALYZE_SYSTEM_PROMPT = """You are a senior code reviewer triaging a pull request for a human-in-the-loop review system. Produce a structured review of the diff.

`confidence` (0.0-1.0) means: how sure you are that your review is complete and correct AND that the PR can be merged without asking the author anything. It is NOT a score of how nice the code is. Be conservative: when unsure, choose the LOWER band.

Work in this order: (1) risk_factors, (2) comments, (3) confidence, (4) confidence_reasoning, which must name the band and every risk factor that lowered the score.

Calibration bands (use the full range; never default to 0.60):
- 0.85-0.95: purely mechanical change (typo, docs, comment, dependency bump, one-line rename/refactor). No behaviour, schema, auth or security impact.
- 0.73-0.84: small, self-contained, tested behaviour change with no open question.
- 0.58-0.72: small feature or data-model/schema change that is probably fine but needs human eyes (edge cases, migration or backward-compatibility question, thin tests). If a migration/existing-data question is unanswered, stay at or below 0.70. A small, focused change whose only concerns are an open migration question and thin tests, with no security red flag, belongs in this band: do not score it below 0.58 for those reasons alone.
- 0.20-0.57: high risk. Touches authentication, passwords, crypto, tokens or secrets, SQL or user-input handling, network/cloud sync, or migrations; contains a security flaw; adds substantial new code without tests; or the intent is unclear.

Security red flags: weak or unsalted password hashing (MD5, SHA1), secrets or tokens stored in plaintext, SQL built with string formatting or concatenation (injection), hard-coded credentials or user ids, plain HTTP for sensitive traffic, disabled TLS verification. One serious red flag caps confidence at 0.55. Two or more cap it at 0.45. New behaviour with no tests lowers the score within its band.

Only report problems you can point to in the diff; give file and line in each comment and use severity blocker/issue/suggestion/nit honestly.

escalation_questions: if confidence < 0.58, give 2-4 specific questions that reference a file or section of the diff and ask about intent, threat model or production configuration (for example why a given algorithm was chosen, or whether a URL is HTTPS in production). Otherwise return an empty list."""

MAX_DIFF_CHARS = 60_000


def _analyze_messages(state) -> list[dict]:
    diff = state["pr_diff"]
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated] ..."
    author = state.get("pr_author") or "unknown"
    files = ", ".join(state.get("pr_files") or [])
    return [
        {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Title: {state['pr_title']}\nAuthor: {author}\nChanged files: {files}\n\nDiff:\n{diff}"
        )},
    ]


SYNTHESIZE_SYSTEM_PROMPT = """You are a senior code reviewer refining an earlier review of a pull request using the human reviewer's answers to your escalation questions.

- Keep every issue the answers confirm (as comments with honest severity).
- Drop or downgrade issues the answers genuinely resolve.
- Add new comments the answers reveal.
- Use the same calibration bands as the initial review: 0.85-0.95 purely mechanical; 0.73-0.84 small, self-contained, tested change; 0.58-0.72 probably fine but needs human eyes; 0.20-0.57 high risk (auth, crypto, secrets, SQL/user input, migrations, security flaws, untested new code, unclear intent).
- Raise `confidence` only where the answers resolve uncertainty. It should usually end higher than the initial value, but stay below 0.85 while any blocker remains.
- Give file and line in each comment. Return `escalation_questions=[]`."""


def _synthesize_messages(state, qa: str) -> list[dict]:
    a = state["analysis"]
    diff = state["pr_diff"]
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated] ..."
    return [
        {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Title: {state['pr_title']}\n\nDiff:\n{diff}\n\n"
            f"Initial analysis:\nSummary: {a.summary}\n"
            f"Risk factors: {a.risk_factors}\n"
            f"Comments: {[c.model_dump() for c in a.comments]}\n"
            f"Confidence: {a.confidence}\nConfidence reasoning: {a.confidence_reasoning}\n\n"
            f"Reviewer Q&A:\n{qa}"
        )},
    ]



# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
        "pr_author": pr.author,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke(_analyze_messages(state))
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision={"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}[decision],
        reason=f"Routed to {decision} (confidence={c:.0%})",
        execution_time_ms=_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    await audit_once(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Awaiting reviewer",
        execution_time_ms=_ms(t0),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer(),
        decision=resp["choice"],
        reason=resp.get("feedback") or "no feedback",
        execution_time_ms=_ms(t0),
    ))
    return {"human_choice": resp["choice"], "human_feedback": resp.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer(),  # commit is only reached after a human answered
        decision={"committed": "approve", "rejected": "reject"}.get(action, "pending"),
        reason=f"final_action={action}",
        execution_time_ms=_ms(t0),
    ))
    return {"final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"Auto-approved without human review; result={action}",
        execution_time_ms=_ms(t0),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    await audit_once(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="escalate",
        reason=f"Awaiting answers to {len(questions)} question(s)",
        execution_time_ms=_ms(t0),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer(),
        decision="escalate",
        reason=f"Reviewer answered {len(answers or {})} question(s)",
        execution_time_ms=_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke(_synthesize_messages(state, qa))
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="escalate",
        reason=refined.confidence_reasoning,
        execution_time_ms=_ms(t0),
    ))
    # node_commit will run next and post the refined review to the PR.
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # An interrupted thread (e.g. the process was killed at a prompt) has pending
        # nodes: continue it from the checkpoint instead of starting over.
        snap = await app.aget_state(cfg)
        first_input = None if snap.next else {"pr_url": pr_url, "thread_id": thread_id}
        result = await app.ainvoke(first_input, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
