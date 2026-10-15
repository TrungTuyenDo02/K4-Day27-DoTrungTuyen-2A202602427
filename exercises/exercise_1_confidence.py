"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
        "pr_author": pr.author,
    }


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



def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke(_analyze_messages(state))
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.comments)} comment(s)")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("escalate", node_escalate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    for terminal in ("auto_approve", "human_approval", "escalate"):
        g.add_edge(terminal, END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
