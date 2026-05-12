"""
debate.py
─────────
Cabinet debate orchestrator for the Kenya AI Executive Roundtable.

Runs a structured, multi-turn policy debate between 7 AI agents representing
Kenya's Cabinet. The President moderates, routes turns, and delivers a final
policy synthesis.

Flow:
    1. President analyses topic → identifies lead ministry → opens debate
    2. Lead CS speaks (RAG-grounded)
    3. Groq routing call → decides next speaker
    4. Repeat until stop condition
    5. President delivers final synthesis

Stop conditions:
    - Max 20 turns reached → President forced synthesis
    - Deadlock: same 2 agents for 4 consecutive turns → President intervenes
    - Convergence detected by router → President synthesises early

Usage:
    python debate.py "Should Kenya extend the SGR to Uganda at a cost of KSh 200 billion?"
    python debate.py --topic "CBC implementation funding" --max-turns 12
    python debate.py --topic "Digital ID rollout" --agents finance ict president
    python debate.py --replay debates/2026-05-11_14-30_sgr_extension.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from groq import Groq

sys.path.insert(0, str(Path(__file__).parent))
from agent import Agent, AGENT_CONFIGS, GROQ_MODEL

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TURNS_DEFAULT   = 20
DEADLOCK_WINDOW     = 4      # consecutive turns of same 2 agents = deadlock
HISTORY_WINDOW      = 8      # turns of history injected into each agent prompt
ROUTER_MAX_TOKENS   = 120
ROUTER_TEMPERATURE  = 0.0
TRANSCRIPT_DIR      = Path(__file__).parent / "debates"

ALL_CS_AGENTS = ["finance", "education", "agriculture", "ict",
                 "infrastructure", "anticorruption"]

# ── Groq client (shared) ──────────────────────────────────────────────────────

def _groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to ~/PRES/Executive/.env")
    return Groq(api_key=api_key)

GROQ = None  # initialised lazily

def groq() -> Groq:
    global GROQ
    if GROQ is None:
        GROQ = _groq_client()
    return GROQ


def groq_call_with_retry(fn, retries: int = 3, delay: float = 4.0):
    """Call a Groq API function with retry on transient network errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            err = str(e)
            # Transient: connection reset, timeout, rate limit
            if any(x in err for x in ["Connection reset", "Connection aborted",
                                       "Transport endpoint", "timed out",
                                       "rate_limit", "503", "502"]):
                wait = delay * (attempt + 1)
                print(f"  [Groq transient error (attempt {attempt+1}/{retries}): retrying in {wait:.0f}s]")
                time.sleep(wait)
                # Re-initialise client on connection errors
                global GROQ
                GROQ = _groq_client()
                continue
            raise  # non-transient — re-raise immediately
    raise last_exc

# ── Routing ───────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are the debate routing engine for Kenya's Cabinet.
Given the current debate topic and the most recent Cabinet Secretary's response,
decide which Cabinet Secretary should speak next.

Cabinet Secretaries available:
  finance        — National Treasury, fiscal policy, debt, tax, budget
  education      — Schools, CBC, TVET, teachers, learning outcomes
  agriculture    — Food security, irrigation, smallholder farmers, KALRO
  ict            — Digital economy, fintech, connectivity, AI, Konza
  infrastructure — Roads, SGR, ports, energy grid, KeNHA
  anticorruption — Governance, procurement, audit, EACC, PPRA

Rules:
- Choose the agent whose mandate is most directly challenged or implicated by the last response.
- Prefer agents who have NOT yet spoken, or who have spoken least.
- If the debate is converging (agents largely agreeing), signal convergence.
- NEVER choose 'president' — the President only speaks when forced or at synthesis.
- Return ONLY valid JSON. No preamble. No explanation outside the JSON.

Return format:
{"next": "<agent_id>", "reason": "<one sentence>", "converging": <true|false>}"""


def route_next_speaker(
    topic: str,
    last_response: str,
    last_speaker: str,
    turn_history: list[str],
    available_agents: list[str],
) -> dict:
    """Call Groq to decide the next speaker. Returns dict with next/reason/converging."""
    history_str = " → ".join(turn_history[-6:]) if turn_history else "none"
    available_str = ", ".join(a for a in available_agents if a != "president")

    user_msg = (
        f"Debate topic: {topic}\n\n"
        f"Last speaker: {last_speaker}\n"
        f"Last response (excerpt): {last_response[:600]}\n\n"
        f"Turn history so far: {history_str}\n"
        f"Available agents: {available_str}\n\n"
        f"Who should speak next?"
    )

    try:
        def _call():
            return groq().chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=ROUTER_MAX_TOKENS,
                temperature=ROUTER_TEMPERATURE,
            )
        resp = groq_call_with_retry(_call)
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        result = json.loads(raw)
        if result.get("next") not in available_agents:
            result["next"] = _fallback_next(turn_history, available_agents)
        return result
    except Exception as e:
        print(f"  [Router error: {e} — using fallback]")
        return {
            "next":       _fallback_next(turn_history, available_agents),
            "reason":     "Router fallback — round-robin",
            "converging": False,
        }


def _fallback_next(turn_history: list[str], available_agents: list[str]) -> str:
    """Round-robin fallback when router fails."""
    cs_agents = [a for a in available_agents if a != "president"]
    if not turn_history:
        return cs_agents[0]
    # Find least-recently-used agent
    for agent in reversed(cs_agents):
        if agent not in turn_history[-len(cs_agents):]:
            return agent
    # All spoken recently — cycle
    last = next((a for a in reversed(turn_history) if a in cs_agents), None)
    if last:
        idx = cs_agents.index(last)
        return cs_agents[(idx + 1) % len(cs_agents)]
    return cs_agents[0]


# ── Opening call — President identifies lead ministry ─────────────────────────

OPENER_SYSTEM = """You are the President of Kenya chairing a Cabinet debate.
Given a policy topic, do THREE things in order:
1. Briefly frame the issue (2-3 sentences).
2. Identify which Cabinet Secretary should lead the debate (one of: finance, education, agriculture, ict, infrastructure, anticorruption).
3. Pose a sharp opening question to that CS.

Return ONLY valid JSON:
{"lead_agent": "<agent_id>", "framing": "<2-3 sentence framing>", "opening_question": "<question to lead CS>"}"""


def president_open(topic: str) -> dict:
    """President analyses topic and produces opening frame + lead agent."""
    try:
        def _call():
            return groq().chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": OPENER_SYSTEM},
                    {"role": "user",   "content": f"Cabinet debate topic: {topic}"},
                ],
                max_tokens=300,
                temperature=0.3,
            )
        resp = groq_call_with_retry(_call)
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        result = json.loads(raw)
        if result.get("lead_agent") not in ALL_CS_AGENTS:
            result["lead_agent"] = "finance"
        return result
    except Exception as e:
        print(f"  [Opener error: {e} — defaulting to finance]")
        return {
            "lead_agent":       "finance",
            "framing":          f"The Cabinet convenes to debate: {topic}",
            "opening_question": f"Cabinet Secretary, please open the debate on: {topic}",
        }


# ── Closing synthesis — President ─────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are the President of Kenya. The Cabinet debate has concluded.
Review the debate transcript below and deliver:
1. A synthesis of the key positions taken (2-3 sentences).
2. Areas of agreement across ministries.
3. Key unresolved tensions.
4. Your final policy directive (what Kenya will do).

Be authoritative and specific. Reference the Cabinet Secretaries by name where relevant.
Do not hedge — make a decision."""


def president_synthesise(topic: str, transcript: list[dict]) -> str:
    """President delivers final synthesis from full debate transcript."""
    # Build a compact transcript string
    debate_text = f"Debate topic: {topic}\n\n"
    for turn in transcript:
        if turn["speaker"] == "president":
            continue
        name = AGENT_CONFIGS[turn["speaker"]]["name"]
        debate_text += f"{name} ({turn['speaker']}):\n{turn['response'][:400]}\n\n"

    try:
        def _call():
            return groq().chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system",  "content": SYNTHESIS_PROMPT},
                    {"role": "user",    "content": debate_text[:6000]},
                ],
                max_tokens=600,
                temperature=0.4,
            )
        resp = groq_call_with_retry(_call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Synthesis error: {e}]"


# ── Deadlock detection ────────────────────────────────────────────────────────

def is_deadlock(turn_history: list[str], window: int = DEADLOCK_WINDOW) -> bool:
    """True if the last `window` turns alternate between exactly 2 agents."""
    if len(turn_history) < window:
        return False
    recent = turn_history[-window:]
    unique = set(recent)
    if len(unique) != 2:
        return False
    # Must strictly alternate
    for i in range(1, len(recent)):
        if recent[i] == recent[i - 1]:
            return False
    return True


# ── Consistency block builder ─────────────────────────────────────────────────

def build_consistency_block(agent_id: str, transcript: list[dict]) -> str:
    """Build a consistency block from this agent's prior turns."""
    prior = [t for t in transcript if t["speaker"] == agent_id]
    if not prior:
        return ""
    block = "YOUR PRIOR POSITIONS IN THIS DEBATE:\n"
    for i, t in enumerate(prior[-3:], 1):  # last 3 turns max
        block += f"  [{i}] {t['response'][:200]}...\n"
    block += "\nMaintain consistency with your prior positions unless new evidence changes your view."
    return block


# ── History builder ───────────────────────────────────────────────────────────

def build_history(transcript: list[dict], window: int = HISTORY_WINDOW) -> list[dict]:
    """Build the last N turns as Groq message history."""
    messages = []
    for turn in transcript[-window:]:
        name = AGENT_CONFIGS.get(turn["speaker"], {}).get("name", turn["speaker"])
        if turn["speaker"] == "president":
            role    = "assistant"
            content = f"[The President]: {turn['response']}"
        else:
            role    = "user"
            content = f"[{name}, {turn['speaker']}]: {turn['response']}"
        messages.append({"role": role, "content": content})
    return messages


# ── Transcript save ───────────────────────────────────────────────────────────

def save_transcript(topic: str, transcript: list[dict], metadata: dict) -> Path:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    slug  = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40].strip("_")
    ts    = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fname = TRANSCRIPT_DIR / f"{ts}_{slug}.json"
    payload = {
        "topic":      topic,
        "timestamp":  ts,
        "metadata":   metadata,
        "transcript": transcript,
    }
    with open(fname, "w") as f:
        json.dump(payload, f, indent=2)
    return fname


# ── Replay ────────────────────────────────────────────────────────────────────

def replay_transcript(path: str):
    """Pretty-print a saved debate transcript."""
    with open(path) as f:
        data = json.load(f)

    _divider("═")
    print(f"  REPLAY: {data['topic']}")
    print(f"  {data['timestamp']}")
    _divider("═")

    for turn in data["transcript"]:
        speaker = turn["speaker"]
        cfg     = AGENT_CONFIGS.get(speaker, {})
        name    = cfg.get("name", speaker)
        _divider()
        print(f"  {name.upper()} ({speaker})")
        _divider()
        print(turn["response"])
        if turn.get("rag_sources"):
            print(f"\n  Sources: {', '.join(turn['rag_sources'])}")
        print()


# ── Print helpers ─────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 70):
    print(char * width)


def _print_turn(turn_num: int, speaker: str, result: dict):
    cfg  = AGENT_CONFIGS.get(speaker, {})
    name = cfg.get("name", speaker)
    _divider()
    print(f"  TURN {turn_num} — {name.upper()} ({speaker})")
    _divider()
    print(result["response"])
    if result.get("rag_sources"):
        print(f"\n  Sources: {', '.join(result['rag_sources'])}")
    print()


# ── Main debate loop ──────────────────────────────────────────────────────────

def run_debate(
    topic:          str,
    max_turns:      int       = MAX_TURNS_DEFAULT,
    active_agents:  list[str] = None,
    verbose:        bool      = False,
    save:           bool      = True,
) -> list[dict]:
    """Run a full cabinet debate. Returns the transcript."""

    active_agents = active_agents or ALL_CS_AGENTS
    # Always include president for routing (not in turn rotation)
    all_participants = active_agents + ["president"]

    transcript:   list[dict] = []
    turn_history: list[str]  = []   # agent_id sequence
    excluded_after_deadlock: set[str] = set()  # agents blocked for 3 turns post-intervention
    exclusion_expires_at: int = 0   # turn number when exclusion lifts
    turn_num = 0

    # ── Pre-load agents ────────────────────────────────────────────────────────
    agents: dict[str, Agent] = {}
    for agent_id in all_participants:
        agents[agent_id] = Agent.from_config(agent_id)

    # ── Opening ────────────────────────────────────────────────────────────────
    _divider("═")
    print(f"  KENYA AI EXECUTIVE ROUNDTABLE")
    print(f"  Topic: {topic}")
    print(f"  Max turns: {max_turns} | Active CS: {', '.join(active_agents)}")
    _divider("═")
    print()

    print("  [President opening...]")
    opening = president_open(topic)
    lead_agent      = opening["lead_agent"]
    framing         = opening["framing"]
    opening_question = opening["opening_question"]

    president_opening_text = (
        f"{framing}\n\n"
        f"I turn first to {AGENT_CONFIGS[lead_agent]['name']}, "
        f"{AGENT_CONFIGS[lead_agent]['title']}.\n\n"
        f"{opening_question}"
    )

    transcript.append({
        "turn":       0,
        "speaker":    "president",
        "response":   president_opening_text,
        "rag_sources": [],
        "thinking":   "",
    })

    _divider("═")
    print(f"  THE PRESIDENT")
    _divider("═")
    print(president_opening_text)
    print()

    # ── Debate loop ────────────────────────────────────────────────────────────
    current_speaker = lead_agent
    first_message   = opening_question

    while turn_num < max_turns:
        turn_num += 1

        print(f"  [Turn {turn_num}: {current_speaker}...]")

        # Build inputs for this turn
        history           = build_history(transcript)
        consistency_block = build_consistency_block(current_speaker, transcript)

        # The message this agent responds to
        if turn_num == 1:
            message = first_message
        else:
            last_turn = transcript[-1]
            last_speaker_name = AGENT_CONFIGS[last_turn["speaker"]]["name"]
            message = (
                f"Debate topic: {topic}\n\n"
                f"{last_speaker_name} just argued: {last_turn['response'][:400]}\n\n"
                f"Do you agree or disagree with this specific position? "
                f"State your stance clearly, cite data from your ministry's mandate, "
                f"and identify one concrete point of disagreement or agreement."
            )

        # Agent speaks
        result = agents[current_speaker].speak(
            message           = message,
            history           = history,
            consistency_block = consistency_block,
            verbose           = verbose,
        )

        turn_record = {
            "turn":       turn_num,
            "speaker":    current_speaker,
            "response":   result["response"],
            "rag_sources": result["rag_sources"],
            "thinking":   result.get("thinking", ""),
        }
        transcript.append(turn_record)
        turn_history.append(current_speaker)

        _print_turn(turn_num, current_speaker, result)

        # ── Stop condition: max turns ──────────────────────────────────────────
        if turn_num >= max_turns:
            print(f"  [Max turns ({max_turns}) reached → President synthesis]")
            break

        # ── Stop condition: deadlock ───────────────────────────────────────────
        if is_deadlock(turn_history):
            locked_pair = set(turn_history[-DEADLOCK_WINDOW:])
            print(f"  [Deadlock detected ({', '.join(locked_pair)}) → President intervenes]")
            intervene_text = (
                f"I note we have been circling between the same positions. "
                f"Let me bring in a fresh perspective. "
                f"What does the rest of Cabinet say?"
            )
            transcript.append({
                "turn":        turn_num + 0.5,
                "speaker":     "president",
                "response":    intervene_text,
                "rag_sources": [],
                "thinking":    "",
            })
            print(f"\n  THE PRESIDENT (intervention): {intervene_text}\n")
            # Exclude the deadlocked pair for the next 3 turns
            excluded_after_deadlock = locked_pair
            exclusion_expires_at    = turn_num + 3
            remaining = [a for a in active_agents if a not in locked_pair]
            if remaining:
                current_speaker = remaining[0]
                continue
            else:
                break

        # ── Route next speaker (respecting post-deadlock exclusion) ───────────
        # Build routing pool — exclude recently deadlocked agents while ban active
        routing_pool = active_agents
        if turn_num < exclusion_expires_at and excluded_after_deadlock:
            routing_pool = [a for a in active_agents if a not in excluded_after_deadlock]
            if not routing_pool:
                routing_pool = active_agents  # safety: never leave empty

        route = route_next_speaker(
            topic            = topic,
            last_response    = result["response"],
            last_speaker     = current_speaker,
            turn_history     = turn_history,
            available_agents = routing_pool,
        )

        if verbose:
            print(f"  [Router → {route['next']} | {route['reason']} | converging={route['converging']}]")

        # ── Stop condition: convergence ────────────────────────────────────────
        if route.get("converging") and turn_num >= 6:
            print(f"  [Convergence detected at turn {turn_num} → President synthesis]")
            break

        current_speaker = route["next"]

    # ── President synthesis ────────────────────────────────────────────────────
    print("  [President synthesising...]")
    synthesis = president_synthesise(topic, transcript)

    transcript.append({
        "turn":       turn_num + 1,
        "speaker":    "president",
        "response":   synthesis,
        "rag_sources": [],
        "thinking":   "",
    })

    _divider("═")
    print("  THE PRESIDENT — FINAL POLICY DIRECTIVE")
    _divider("═")
    print(synthesis)
    print()

    # ── Save transcript ────────────────────────────────────────────────────────
    if save:
        metadata = {
            "max_turns":     max_turns,
            "turns_run":     turn_num,
            "active_agents": active_agents,
            "lead_agent":    lead_agent,
        }
        path = save_transcript(topic, transcript, metadata)
        _divider("═")
        print(f"  Transcript saved → {path}")
        _divider("═")

    return transcript


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kenya AI Executive Roundtable — Cabinet debate simulator"
    )
    parser.add_argument(
        "topic_pos", nargs="?",
        default=None,
        metavar="TOPIC",
        help="Debate topic as positional argument (wrap in quotes)",
    )
    parser.add_argument(
        "--topic", "-t",
        default=None,
        help="Debate topic (flag form)",
    )
    parser.add_argument(
        "--max-turns", "-m",
        type=int,
        default=MAX_TURNS_DEFAULT,
        help=f"Maximum debate turns (default: {MAX_TURNS_DEFAULT})",
    )
    parser.add_argument(
        "--agents", "-a",
        nargs="+",
        choices=ALL_CS_AGENTS,
        default=None,
        help="Agents to include (default: all 6 CS agents)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieval and routing debug info",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save transcript to disk",
    )
    parser.add_argument(
        "--replay", "-r",
        default=None,
        help="Replay a saved transcript JSON file",
    )
    args = parser.parse_args()

    # ── Replay mode ────────────────────────────────────────────────────────────
    if args.replay:
        replay_transcript(args.replay)
        sys.exit(0)

    # ── Debate mode ────────────────────────────────────────────────────────────
    # Resolve topic: --topic flag takes precedence, then positional, then default
    DEFAULT_TOPIC = (
        "The Infrastructure CS is proposing a KSh 200 billion SGR extension "
        "to Uganda financed through a new sovereign bond. Should Kenya proceed?"
    )
    final_topic = args.topic or getattr(args, 'topic_pos', None) or DEFAULT_TOPIC

    run_debate(
        topic         = final_topic,
        max_turns     = args.max_turns,
        active_agents = args.agents,
        verbose       = args.verbose,
        save          = not args.no_save,
    )