"""
rubric.py
─────────────────────────────────────────────────────────────────────────────
Evaluation rubric for Kenya AI Executive Roundtable agents.
Scores a single agent turn across four dimensions.

Usage:
    python rubric.py <agent_id> "<message>"
    python rubric.py finance "What is Kenya's current debt-to-GDP ratio?"
    python rubric.py agriculture "Should Kenya expand irrigation funding in FY2025-26?"
    python rubric.py finance "What is the debt outlook?" --output result.json

Dimensions:
    KGS  Kenya Grounding Score      — claims traceable to retrieved chunks
    RDS  Reasoning Depth Score      — second-order consequences and causal chains
    ICS  Internal Consistency Score — no contradiction of prior positions
    OPS  Outcome Projection Score   — measurable targets, timelines, indicators

Each dimension scores 0.0 – 1.0. Final score is the mean of all four.
"""

import os
import sys
import json
import argparse
import textwrap

from groq import Groq

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.expanduser("~/PRES/Executive/Finance"))

from agent import Agent
from retriever import retrieve, format_context

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_MODEL  = "llama-3.3-70b-versatile"
TEMPERATURE = 0.0   # deterministic scoring — never change this
MAX_TOKENS  = 120   # score + one-line reason only

# ── Scoring prompts ───────────────────────────────────────────────────────────

_KGS_PROMPT = """You are a policy debate evaluator scoring one agent response.

SCORING CRITERION — Kenya Grounding Score (KGS):
Does the agent's response make claims supported by the provided retrieved chunks?
A high score means specific figures, facts, or arguments are traceable to the chunks.
A low score means the agent ignored the chunks or introduced facts not present in them.

SCALE:
1.0 = All major claims directly supported by the retrieved chunks
0.5 = Some claims grounded in chunks, others vague or unsupported
0.0 = Response contradicts, ignores, or fabricates beyond the chunks

RETRIEVED CHUNKS (this is what the agent was given):
{chunks}

AGENT RESPONSE:
{response}

Return ONLY valid JSON, no preamble, no markdown:
{{"score": <float 0.0-1.0>, "reason": "<one sentence max>"}}"""


_RDS_PROMPT = """You are a policy debate evaluator scoring one agent response.

SCORING CRITERION — Reasoning Depth Score (RDS):
Does the agent reason through second-order consequences?
Look for explicit causal chains: "if X then Y, which leads to Z."
Look for tradeoffs acknowledged, counter-arguments addressed, conditional logic.

SCALE:
1.0 = Explicit causal chain with at least one second-order consequence identified
0.5 = Single-level reasoning — states a cause or effect but does not chain further
0.0 = Assertion only — no reasoning, no consequences, no tradeoffs

AGENT RESPONSE:
{response}

Return ONLY valid JSON, no preamble, no markdown:
{{"score": <float 0.0-1.0>, "reason": "<one sentence max>"}}"""


_ICS_PROMPT = """You are a policy debate evaluator scoring one agent response.

SCORING CRITERION — Internal Consistency Score (ICS):
Does the agent's response contradict any of their prior stated positions?
If there are no prior positions, score 1.0 automatically.

SCALE:
1.0 = No contradiction of prior positions, or no prior positions exist
0.5 = Mild shift in position without acknowledgment
0.0 = Direct contradiction of a prior stated position

AGENT'S PRIOR POSITIONS (consistency block):
{consistency_block}

AGENT RESPONSE (current turn):
{response}

Return ONLY valid JSON, no preamble, no markdown:
{{"score": <float 0.0-1.0>, "reason": "<one sentence max>"}}"""


_OPS_PROMPT = """You are a policy debate evaluator scoring one agent response.

SCORING CRITERION — Outcome Projection Score (OPS):
Does the agent's proposal include measurable, specific outcomes?
Look for: a specific figure (KES amount, percentage), a timeline or fiscal year,
and an economic or social indicator it affects.

SCALE:
1.0 = Specific figure + timeline + named indicator
      e.g. "KES 4.2B by FY2026-27 reducing import dependency by 12%"
0.5 = Specific figure but missing timeline or indicator
0.0 = No measurable projection at all — vague recommendation only

AGENT RESPONSE:
{response}

Return ONLY valid JSON, no preamble, no markdown:
{{"score": <float 0.0-1.0>, "reason": "<one sentence max>"}}"""


# ── Judge caller ──────────────────────────────────────────────────────────────

def _call_judge(prompt: str, client: Groq) -> dict:
    """Send a scoring prompt to Groq and parse the JSON response."""
    try:
        resp = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"score": 0.0, "reason": f"[scoring error: {e}]"}


# ── Per-dimension scorers ─────────────────────────────────────────────────────

def score_kgs(response: str, chunks: list, client: Groq) -> dict:
    chunk_text = format_context(chunks) if chunks else "No chunks retrieved."
    return _call_judge(_KGS_PROMPT.format(chunks=chunk_text, response=response), client)

def score_rds(response: str, client: Groq) -> dict:
    return _call_judge(_RDS_PROMPT.format(response=response), client)

def score_ics(response: str, consistency_block: str, client: Groq) -> dict:
    block = consistency_block.strip() if consistency_block else "None — this is the first turn."
    return _call_judge(_ICS_PROMPT.format(consistency_block=block, response=response), client)

def score_ops(response: str, client: Groq) -> dict:
    return _call_judge(_OPS_PROMPT.format(response=response), client)


# ── Full turn scorer ──────────────────────────────────────────────────────────

def score_turn(
    response:          str,
    chunks:            list,
    consistency_block: str,
    client:            Groq,
    verbose:           bool = False,
) -> dict:
    """Run all four scoring dimensions on a single agent turn."""
    if verbose: print("\n[Rubric] Scoring KGS...")
    kgs = score_kgs(response, chunks, client)

    if verbose: print("[Rubric] Scoring RDS...")
    rds = score_rds(response, client)

    if verbose: print("[Rubric] Scoring ICS...")
    ics = score_ics(response, consistency_block, client)

    if verbose: print("[Rubric] Scoring OPS...")
    ops = score_ops(response, client)

    mean = round((kgs["score"] + rds["score"] + ics["score"] + ops["score"]) / 4, 3)

    return {"KGS": kgs, "RDS": rds, "ICS": ics, "OPS": ops, "mean": mean}


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(result: dict, scores: dict, width: int = 72):
    print("\n" + "═" * width)
    print(f"  {result['name']}  ·  {result['speaker']}")
    print("═" * width)

    # Response
    print("\n[Response]\n")
    for line in result["response"].split("\n"):
        print(textwrap.fill(line, width=width) if line.strip() else "")

    # Thinking block (DeepSeek R1 style — only shown if present)
    if result.get("thinking"):
        print("\n[Thinking — first 300 chars]\n")
        print(textwrap.fill(result["thinking"][:300] + "...", width=width))

    # RAG sources
    if result.get("rag_sources"):
        print("\n[RAG Sources]")
        for s in result["rag_sources"]:
            print(f"  · {s}")

    # Score table
    print("\n" + "─" * width)
    print("  RUBRIC SCORES\n")

    dims = [
        ("KGS", "Kenya Grounding"),
        ("RDS", "Reasoning Depth"),
        ("ICS", "Internal Consistency"),
        ("OPS", "Outcome Projection"),
    ]
    for code, label in dims:
        s     = scores[code]
        score = s["score"]
        filled = int(score * 10)
        bar   = "█" * filled + "░" * (10 - filled)
        print(f"  {code}  {bar}  {score:.1f}  {label}")
        reason = s.get("reason", "")
        if reason:
            wrapped = textwrap.fill(
                reason,
                width            = width - 7,
                subsequent_indent= " " * 7,
            )
            print(f"       {wrapped}")
        print()

    mean   = scores["mean"]
    filled = int(mean * 10)
    mbar   = "█" * filled + "░" * (10 - filled)
    print(f"  MEAN {mbar}  {mean:.3f}")
    print("─" * width + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score a Kenya AI Executive Roundtable agent turn on four rubric dimensions."
    )
    parser.add_argument(
        "agent_id", nargs="?", default="finance",
        help="Agent ID: finance | education | agriculture | ict | infrastructure | anticorruption | president",
    )
    parser.add_argument(
        "message", nargs="?",
        default="What is Kenya's current debt-to-GDP ratio and what are the fiscal risks going into FY2025-26?",
        help="The debate prompt or message to send to the agent",
    )
    parser.add_argument(
        "--consistency", default="",
        help="Agent's prior positions this debate (paste consistency block as string)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the full JSON result (e.g. result.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show retrieval and scoring debug output",
    )
    args = parser.parse_args()

    # Validate environment
    for key in ("GROQ_API_KEY", "VOYAGE_API_KEY"):
        if not os.environ.get(key):
            print(f"[Error] Missing environment variable: {key}")
            sys.exit(1)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print(f"\nAgent  : {args.agent_id}")
    print(f"Message: {args.message}")

    # Run agent
    agent  = Agent.from_config(args.agent_id)
    result = agent.speak(
        args.message,
        consistency_block = args.consistency,
        verbose           = args.verbose,
    )

    # Retrieve chunks for KGS scoring
    # Note: agent.speak() retrieves internally too — this second call gives
    # the scorer access to raw chunk text for grounding verification.
    # Future improvement: add rag_chunks to agent.speak()'s return dict.
    chunks = retrieve(args.message, agent=args.agent_id, verbose=False)

    # Score all four dimensions
    scores = score_turn(
        response          = result["response"],
        chunks            = chunks,
        consistency_block = args.consistency,
        client            = client,
        verbose           = args.verbose,
    )

    # Print to terminal
    print_results(result, scores)

    # Optionally save JSON
    if args.output:
        payload = {
            "agent_id":          args.agent_id,
            "message":           args.message,
            "response":          result["response"],
            "thinking":          result.get("thinking", ""),
            "rag_sources":       result.get("rag_sources", []),
            "consistency_block": args.consistency,
            "scores":            scores,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Saved] {args.output}\n")


if __name__ == "__main__":
    main()