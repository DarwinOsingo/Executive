"""
agent.py
────────
Inference wrapper for Kenya AI Executive Roundtable agents.
Combines Groq LLM inference with RAG context retrieval.

Template for all 7 cabinet agents — same code, different persona and RAG filter.
Agent filtering uses agent_access[] array in Qdrant payloads (multi-agent corpus).

Current inference: Groq (llama-3.3-70b-versatile)
Post fine-tuning:  Together AI (DeepSeek R1 Distill Qwen 14B + LoRA adapter)

Usage (single agent):
    python agent.py "The Infrastructure CS is proposing a KSh 50B SGR extension."
    python agent.py --agent education "What is Kenya's CBC implementation status?"

Usage (stress test — all agents, multiple queries):
    python agent.py --stress-test
    python agent.py --stress-test --rag-only   # retrieval only, no LLM call
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from same directory as this script

load_dotenv(Path(__file__).parent / ".env", override=True)

from groq import Groq

# Retriever lives at the Executive root — same level as agent.py
sys.path.insert(0, str(Path(__file__).parent))
from retriever import retrieve, format_context

# ── Agent configs ─────────────────────────────────────────────────────────────

AGENT_CONFIGS = {
    "finance": {
        "name":     "Prof. Kamau",
        "title":    "Cabinet Secretary for the National Treasury and Economic Planning",
        "agent_id": "finance",
        "system_prompt": (
            "You are Kenya's Cabinet Secretary for the National Treasury and Economic Planning. "
            "You are a disciplined, data-driven economist with deep expertise in Kenya's fiscal policy, "
            "public debt management, revenue mobilization, budget allocation, and macroeconomic stability. "
            "You speak with authority grounded in Kenya's own government data. You cite specific figures, "
            "budget lines, and economic indicators. You challenge proposals that are fiscally irresponsible "
            "and champion reforms that move Kenya toward upper-middle income status by 2030. "
            "You are not a politician — you are a technocrat who follows the numbers. "
            "When presented with quantitative scenarios, you close the accounting loop explicitly: "
            "Debt_t = Debt_{t-1} + Deficit_t and GDP_t = GDP_{t-1} × (1 + nominal growth rate). "
            "You validate projected debt ratios mechanically before drawing conclusions."
        ),
    },
    "education": {
        "name":     "Dr. Njeri",
        "title":    "Cabinet Secretary for Education",
        "agent_id": "education",
        "system_prompt": (
            "You are Kenya's Cabinet Secretary for Education. You are a passionate advocate for human "
            "capital development, with deep expertise in Kenya's education system, CBC implementation, "
            "TVET reform, curriculum development, and teacher policy. You ground your arguments in "
            "enrollment data, learning outcomes, pupil-teacher ratios, and long-term productivity returns. "
            "You push back against budget cuts that undermine Kenya's human capital pipeline and champion "
            "evidence-based education reform. You cite KNEC examination statistics, TSC workforce data, "
            "and education sector expenditure figures."
        ),
    },
    "agriculture": {
        "name":     "Dr. Achieng",
        "title":    "Cabinet Secretary for Agriculture and Livestock Development",
        "agent_id": "agriculture",
        "system_prompt": (
            "You are Kenya's Cabinet Secretary for Agriculture and Livestock Development. You are a "
            "pragmatic, field-grounded agricultural economist with deep expertise in Kenya's food security, "
            "irrigation infrastructure, smallholder farmer support, and agricultural value chains. "
            "You speak in terms of yield data, rainfall patterns, and market access. You defend food "
            "security as a national security issue and push for investment in irrigation and post-harvest "
            "infrastructure. You cite KALRO research data, AFA production statistics, and FSRP outcomes."
        ),
    },
    "ict": {
        "name":     "Eng. Mwangi",
        "title":    "Cabinet Secretary for ICT and Digital Economy",
        "agent_id": "ict",
        "system_prompt": (
            "You are Kenya's Cabinet Secretary for ICT and the Digital Economy. You are an optimistic, "
            "evidence-driven technologist with deep expertise in Kenya's digital infrastructure, fintech "
            "ecosystem, Silicon Savannah, and digital public goods. You argue that digital transformation "
            "is the fastest path to upper-middle income status and push for investment in connectivity, "
            "digital ID, and e-government. You cite mobile money penetration, internet access statistics, "
            "CA regulatory data, and Kenya AI Strategy targets."
        ),
    },
    "infrastructure": {
        "name":     "Eng. Otieno",
        "title":    "Cabinet Secretary for Infrastructure",
        "agent_id": "infrastructure",
        "system_prompt": (
            "You are Kenya's Cabinet Secretary for Infrastructure. You are a no-nonsense civil engineer "
            "turned policymaker with deep expertise in Kenya's roads, ports, energy grid, and SGR corridor. "
            "You argue in terms of cost-benefit ratios, traffic volumes, and economic multipliers. "
            "You push for infrastructure investment as the foundation of economic growth but are pragmatic "
            "about financing — you will not defend projects that cannot demonstrate a credible return. "
            "You cite KeNHA road condition data, EPRA energy statistics, and KPA port throughput figures."
        ),
    },
    "anticorruption": {
        "name":     "Justice Waweru",
        "title":    "Cabinet Secretary for Public Service and Anti-Corruption",
        "agent_id": "anticorruption",
        "system_prompt": (
            "You are Kenya's Cabinet Secretary responsible for governance, rule of law, and anti-corruption. "
            "You are a former judge — precise, principled, and uncompromising on institutional integrity. "
            "You cite Auditor-General findings, EACC reports, procurement violations, and governance indices. "
            "You challenge any proposal that creates opportunities for rent-seeking or weakens accountability "
            "mechanisms. You believe no development agenda survives systemic corruption and you make that "
            "case with documented evidence from PPRA reports and MER assessments."
        ),
    },
    "president": {
        "name":     "The President",
        "title":    "President of the Republic of Kenya",
        "agent_id": "president",
        "system_prompt": (
            "You are the President of the Republic of Kenya, chairing a Cabinet debate on national policy. "
            "You are the final policy authority. Your role is to moderate debate, identify which Cabinet "
            "Secretary is best placed to address each issue, synthesise competing positions, and drive "
            "toward actionable policy decisions. You are politically astute but ultimately guided by what "
            "is best for Kenya's development trajectory. You do not take detailed technical positions — "
            "you ask the right questions, direct debate to the right experts, and make final calls when "
            "consensus cannot be reached."
        ),
    },
}

# ── Groq config ───────────────────────────────────────────────────────────────

GROQ_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS  = 1024
TEMPERATURE = 0.7

# ── Stress test queries — one per agent, designed to exercise the RAG ─────────

STRESS_TEST_QUERIES = {
    "finance": [
        "What was Kenya's fiscal deficit as a percentage of GDP in FY 2022/23 and what drove it?",
        "What is Kenya's current debt-to-GDP ratio and is it within the statutory ceiling?",
        "How did KRA perform against its revenue target in the most recent financial year?",
    ],
    "education": [
        "What is Kenya's current primary school net enrollment rate and how has it trended?",
        "What are the key findings from the most recent KCSE examination statistics report?",
        "How many teachers does the TSC employ and what is the current teacher-to-pupil ratio?",
    ],
    "agriculture": [
        "What was Kenya's national food production output in the most recent agricultural year?",
        "What is the status of irrigation coverage in Kenya and what are the targets?",
        "What are the key findings from KALRO's most recent annual research report?",
    ],
    "ict": [
        "What is Kenya's current mobile money penetration rate and internet access coverage?",
        "What are the key targets in Kenya's National ICT Masterplan?",
        "What is the status of the Konza Technopolis development project?",
    ],
    "infrastructure": [
        "What is the current condition of Kenya's classified road network?",
        "What is Kenya's current installed electricity generation capacity?",
        "What were the findings of the most recent EPRA energy statistics report?",
    ],
    "anticorruption": [
        "What were the key findings of the Auditor General's report on national government for 2022/23?",
        "What is Kenya's score on the FATF mutual evaluation and what are the key risk areas?",
        "How many procurement irregularities did PPRA identify in the most recent reporting period?",
    ],
    "president": [
        "What is the status of Kenya's devolution framework and county fiscal performance?",
        "What are Kenya's key targets under the Bottom-Up Economic Transformation Agenda?",
        "What cross-cutting risks does the IMF identify for Kenya's economic outlook?",
    ],
}


# ── Agent class ───────────────────────────────────────────────────────────────

@dataclass
class Agent:
    name:          str
    title:         str
    agent_id:      str
    system_prompt: str
    _client:       Optional[Groq] = field(default=None, repr=False)

    def __post_init__(self):
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to ~/PRES/Executive/.env"
            )
        self._client = Groq(api_key=api_key)

    @classmethod
    def from_config(cls, agent_id: str) -> "Agent":
        if agent_id not in AGENT_CONFIGS:
            raise ValueError(
                f"Unknown agent: '{agent_id}'. "
                f"Choose from: {list(AGENT_CONFIGS.keys())}"
            )
        cfg = AGENT_CONFIGS[agent_id]
        return cls(
            name          = cfg["name"],
            title         = cfg["title"],
            agent_id      = cfg["agent_id"],
            system_prompt = cfg["system_prompt"],
        )

    def _build_system_prompt(self, context: str) -> str:
        if not context:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            f"RELEVANT CONTEXT FROM KENYA GOVERNMENT DOCUMENTS:\n"
            f"{context}\n\n"
            f"Ground your response in the above documents where relevant. "
            f"Cite sources by filename and page number when making specific claims."
        )

    def speak(
        self,
        message:           str,
        history:           list[dict] = None,
        consistency_block: str        = "",
        verbose:           bool       = False,
    ) -> dict:
        """Generate a grounded response from this agent.

        Returns dict with:
            speaker, name, response, thinking, rag_sources, rag_chunks
        """
        history = history or []

        # 1. Retrieve RAG context
        chunks  = retrieve(message, agent=self.agent_id, verbose=verbose)
        context = format_context(chunks)
        sources = [
            f"{c.get('source_file')} p.{c.get('page_number')}"
            for c in chunks
        ]

        # 2. Build system prompt with injected RAG context
        system = self._build_system_prompt(context)

        # 3. Inject consistency block if present
        user_message = message
        if consistency_block:
            user_message = f"{consistency_block}\n\n---\n\n{message}"

        # 4. Build messages
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # 5. Call Groq
        response = self._client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )

        raw = response.choices[0].message.content
        thinking, answer = _parse_thinking(raw)

        return {
            "speaker":     self.agent_id,
            "name":        self.name,
            "response":    answer,
            "thinking":    thinking,
            "rag_sources": sources,
            "rag_chunks":  chunks,
        }

    def retrieve_only(self, message: str, verbose: bool = False) -> dict:
        """Run retrieval only — no LLM call. For RAG stress testing."""
        chunks  = retrieve(message, agent=self.agent_id, verbose=verbose)
        sources = [
            f"{c.get('source_file')} p.{c.get('page_number')}"
            for c in chunks
        ]
        return {
            "speaker":    self.agent_id,
            "name":       self.name,
            "rag_sources": sources,
            "rag_chunks":  chunks,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_thinking(text: str) -> tuple[str, str]:
    """Separate <think>...</think> from actual response (DeepSeek R1 style)."""
    import re
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip(), text[match.end():].strip()
    return "", text.strip()


def _print_divider(char: str = "─", width: int = 70):
    print(char * width)


def _print_rag_result(result: dict, query: str, query_idx: int):
    """Pretty-print retrieval results for stress test."""
    chunks = result.get("rag_chunks", [])
    print(f"\n  Query {query_idx}: {query[:80]}{'...' if len(query) > 80 else ''}")
    if not chunks:
        print("  ⚠️  NO CHUNKS RETURNED — retrieval failed")
        return
    for i, c in enumerate(chunks, 1):
        fy     = c.get("fiscal_year", "?")
        pri    = c.get("priority", "?")
        dom    = c.get("domain", "?")
        src    = c.get("source_file", "?")
        page   = c.get("page_number", "?")
        hdg    = (c.get("heading_path") or "")[:50]
        print(f"    [{i}] {src} p.{page} | FY:{fy} | {pri} | {dom}")
        print(f"         {hdg}")


# ── Stress test ───────────────────────────────────────────────────────────────

def run_stress_test(rag_only: bool = False, agents_to_test: list = None):
    """Run stress test across all (or selected) agents.

    For each agent runs its STRESS_TEST_QUERIES and reports:
    - Whether retrieval returned chunks (pass/fail)
    - Source documents retrieved
    - Domain and priority of returned chunks
    - (If not rag_only) LLM response quality
    """
    agents_to_test = agents_to_test or list(AGENT_CONFIGS.keys())

    print("\n" + "═" * 70)
    print("  KENYA AI EXECUTIVE ROUNDTABLE — RAG STRESS TEST")
    print(f"  Mode: {'RAG retrieval only' if rag_only else 'Full pipeline (RAG + LLM)'}")
    print(f"  Agents: {', '.join(agents_to_test)}")
    print("═" * 70)

    results_summary = {}

    for agent_id in agents_to_test:
        cfg     = AGENT_CONFIGS[agent_id]
        agent   = Agent.from_config(agent_id)
        queries = STRESS_TEST_QUERIES[agent_id]

        print(f"\n{'─' * 70}")
        print(f"  {cfg['name']} — {cfg['title']}")
        print(f"{'─' * 70}")

        agent_results = {"pass": 0, "fail": 0, "queries": []}

        for qi, query in enumerate(queries, 1):
            t0 = time.time()

            if rag_only:
                result = agent.retrieve_only(query, verbose=False)
            else:
                result = agent.speak(query, verbose=False)

            elapsed = time.time() - t0
            chunks  = result.get("rag_chunks", [])
            passed  = len(chunks) == 3

            _print_rag_result(result, query, qi)
            print(f"         ⏱  {elapsed:.1f}s  {'✓ 3 chunks' if passed else f'⚠ {len(chunks)} chunks'}")

            if not rag_only and result.get("response"):
                resp_preview = result["response"][:200].replace("\n", " ")
                print(f"\n  Response preview:")
                print(f"    {resp_preview}...")
                if result.get("rag_sources"):
                    print(f"  Sources: {', '.join(result['rag_sources'])}")

            agent_results["pass" if passed else "fail"] += 1
            agent_results["queries"].append({
                "query":   query,
                "passed":  passed,
                "chunks":  len(chunks),
                "elapsed": elapsed,
            })

        results_summary[agent_id] = agent_results
        print(
            f"\n  {agent_id}: "
            f"{agent_results['pass']}/{len(queries)} queries returned 3 chunks"
        )

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  STRESS TEST SUMMARY")
    print("═" * 70)
    total_pass = total_fail = 0
    for agent_id, r in results_summary.items():
        p, f = r["pass"], r["fail"]
        total_pass += p
        total_fail += f
        status = "✓" if f == 0 else "⚠️ "
        print(f"  {status}  {agent_id:<20} {p}/{p+f} queries passed")

    print(f"\n  Total: {total_pass}/{total_pass + total_fail} queries returned 3 chunks")
    if total_fail == 0:
        print("  ✓ All retrieval paths working correctly")
    else:
        print(f"  ⚠️  {total_fail} queries returned fewer than 3 chunks — investigate")
    print("═" * 70 + "\n")

    return results_summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kenya AI Executive Roundtable — agent inference + RAG stress test"
    )
    parser.add_argument(
        "message", nargs="?",
        default=None,
        help="Message to send to the agent",
    )
    parser.add_argument(
        "--agent", "-a",
        default="finance",
        choices=list(AGENT_CONFIGS.keys()),
        help="Agent to use (default: finance)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieval debug info",
    )
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Run full RAG stress test across all agents",
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Stress test: retrieval only, no LLM call",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENT_CONFIGS.keys()),
        default=None,
        help="Agents to include in stress test (default: all)",
    )
    args = parser.parse_args()

    # ── Stress test mode ───────────────────────────────────────────────────────
    if args.stress_test:
        run_stress_test(rag_only=args.rag_only, agents_to_test=args.agents)
        sys.exit(0)

    # ── Single agent mode ──────────────────────────────────────────────────────
    message = args.message or (
        "The Infrastructure CS is proposing a KSh 50 billion SGR extension "
        "to be financed through a new sovereign bond. What is your position?"
    )

    print(f"\nAgent  : {args.agent}")
    print(f"Message: {message}\n")

    agent  = Agent.from_config(args.agent)
    result = agent.speak(message, verbose=args.verbose)

    cfg = AGENT_CONFIGS[args.agent]
    print(f"\n── {result['name']} ──\n")
    if result["thinking"]:
        print(f"[Thinking]\n{result['thinking'][:400]}...\n")
    print(f"[Response]\n{result['response']}")
    print(f"\n[Sources]")
    for src in result["rag_sources"]:
        print(f"  {src}")