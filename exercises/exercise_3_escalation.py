"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence is below ESCALATE_THRESHOLD, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha, "pr_author": pr.author}


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



def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke(_analyze_messages(state))
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        # fallback when the LLM didn't generate any questions
        questions = ["What is the intent of this PR?", "Any migration concerns?"]

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    console.print("[cyan]→ synthesize[/cyan]")
    a = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {ans}" for q, ans in (state.get("escalation_answers") or {}).items())
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = llm.invoke(_synthesize_messages(state, qa))
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


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


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
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
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
