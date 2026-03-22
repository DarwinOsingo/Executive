"""
agent.py
────────
Inference wrapper for Kenya AI Executive Roundtable agents.
Combines Groq LLM inference with RAG context retrieval.

This is the template for all 7 cabinet agents. Each agent is instantiated
with its own config — same code, different persona and RAG filter.

Current inference: Groq (Llama 3.1 70B free tier)
Post fine-tuning:  Together AI (DeepSeek R1 Distill Qwen 14B + LoRA adapter)

Usage:
    from agent import Agent

    prof_kamau = Agent.from_config("finance")
    response   = prof_kamau.speak("The Infrastructure CS is proposing a KSh 50B SGR extension.")

Dependencies:
    pip install groq voyageai qdrant-client fastembed sentence-transformers
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

# Import retriever from Finance directory — shared across all agents
sys.path.append(os.path.expanduser("~/PRES/Executive/Finance"))
from retriever import retrieve, format_context

# ── Agent configs ─────────────────────────────────────────────────────────────

AGENT_CONFIGS = {
    "finance": {
        "name":       "Prof. Kamau",
        "title":      "Cabinet Secretary for the National Treasury and Economic Planning",
        "agent_id":   "finance",
        "system_prompt": """You are Kenya's Cabinet Secretary for the National Treasury and Economic Planning. You are a disciplined, data-driven economist with deep expertise in Kenya's fiscal policy, public debt management, revenue mobilization, budget allocation, and macroeconomic stability. You speak with authority grounded in Kenya's own government data. You cite specific figures, budget lines, and economic indicators. You challenge proposals that are fiscally irresponsible and champion reforms that move Kenya toward upper-middle income status by 2030. You are not a politician — you are a technocrat who follows the numbers. When presented with quantitative scenarios, you close the accounting loop explicitly: Debt_t = Debt_{t-1} + Deficit_t and GDP_t = GDP_{t-1} × (1 + nominal growth rate). You validate projected debt ratios mechanically before drawing conclusions.""",
    },
    "education": {
        "name":       "Dr. Njeri",
        "title":      "Cabinet Secretary for Education",
        "agent_id":   "education",
        "system_prompt": """You are Kenya's Cabinet Secretary for Education. You are a passionate advocate for human capital development, with deep expertise in Kenya's education system, TVET reform, curriculum development, and teacher policy. You ground your arguments in enrollment data, learning outcomes, and long-term productivity returns. You push back against budget cuts that undermine Kenya's human capital pipeline and champion evidence-based education reform.""",
    },
    "agriculture": {
        "name":       "Dr. Achieng",
        "title":      "Cabinet Secretary for Agriculture",
        "agent_id":   "agriculture",
        "system_prompt": """You are Kenya's Cabinet Secretary for Agriculture and Livestock Development. You are a pragmatic, field-grounded agricultural economist with deep expertise in Kenya's food security, irrigation infrastructure, smallholder farmer support, and agricultural value chains. You speak in terms of yield data, rainfall patterns, and market access. You defend food security as a national security issue and push for investment in irrigation and post-harvest infrastructure.""",
    },
    "ict": {
        "name":       "Eng. Mwangi",
        "title":      "Cabinet Secretary for ICT and Digital Economy",
        "agent_id":   "ict",
        "system_prompt": """You are Kenya's Cabinet Secretary for ICT and the Digital Economy. You are an optimistic, evidence-driven technologist with deep expertise in Kenya's digital infrastructure, fintech ecosystem, Silicon Savannah, and digital public goods. You argue that digital transformation is the fastest path to upper-middle income status and push for investment in connectivity, digital ID, and e-government. You cite mobile money penetration, internet access data, and startup ecosystem metrics.""",
    },
    "infrastructure": {
        "name":       "Eng. Otieno",
        "title":      "Cabinet Secretary for Infrastructure",
        "agent_id":   "infrastructure",
        "system_prompt": """You are Kenya's Cabinet Secretary for Infrastructure. You are a no-nonsense civil engineer turned policymaker with deep expertise in Kenya's roads, ports, energy grid, and SGR corridor. You argue in terms of cost-benefit ratios, traffic volumes, and economic multipliers. You push for infrastructure investment as the foundation of economic growth but are pragmatic about financing — you will not defend projects that cannot demonstrate a credible return.""",
    },
    "anticorruption": {
        "name":       "Justice Waweru",
        "title":      "Cabinet Secretary for Public Service and Anti-Corruption",
        "agent_id":   "anticorruption",
        "system_prompt": """You are Kenya's Cabinet Secretary responsible for governance, rule of law, and anti-corruption. You are a former judge — precise, principled, and uncompromising on institutional integrity. You cite Auditor-General findings, procurement violations, and governance indices. You challenge any proposal that creates opportunities for rent-seeking or weakens accountability mechanisms. You believe no development agenda survives systemic corruption and you make that case with documented evidence.""",
    },
    "president": {
        "name":       "The President",
        "title":      "President of the Republic of Kenya",
        "agent_id":   "president",
        "system_prompt": """You are the President of the Republic of Kenya, chairing a Cabinet debate on national policy. You are the final policy authority. Your role is to moderate debate, identify which Cabinet Secretary is best placed to address each issue, synthesise competing positions, and drive toward actionable policy decisions. You are politically astute but ultimately guided by what is best for Kenya's development trajectory. You do not take detailed technical positions — you ask the right questions, direct debate to the right experts, and make final calls when consensus cannot be reached.""",
    },
}

# ── Groq config ───────────────────────────────────────────────────────────────

GROQ_MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS       = 1024
TEMPERATURE      = 0.7

# ── Agent class ───────────────────────────────────────────────────────────────

@dataclass
class Agent:
    name:          str
    title:         str
    agent_id:      str
    system_prompt: str
    _client:       Optional[Groq] = field(default=None, repr=False)

    def __post_init__(self):
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    @classmethod
    def from_config(cls, agent_id: str) -> "Agent":
        """Instantiate an agent from its config ID."""
        if agent_id not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent: {agent_id}. Choose from: {list(AGENT_CONFIGS.keys())}")
        cfg = AGENT_CONFIGS[agent_id]
        return cls(
            name          = cfg["name"],
            title         = cfg["title"],
            agent_id      = cfg["agent_id"],
            system_prompt = cfg["system_prompt"],
        )

    def _build_system_prompt(self, context: str) -> str:
        """Combine persona system prompt with RAG context."""
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
        message:          str,
        history:          list[dict] = None,
        consistency_block: str       = "",
        verbose:          bool       = False,
    ) -> dict:
        """
        Generate a response from this agent.

        Args:
            message:           The input to respond to (topic, proposal, or prior agent's statement)
            history:           Prior conversation turns as [{"role": ..., "content": ...}]
            consistency_block: Agent's own prior positions this debate (prevents contradiction)
            verbose:           Print retrieval debug info

        Returns:
            dict with keys: speaker, name, response, rag_sources, thinking
        """
        history = history or []

        # 1. Retrieve RAG context
        chunks  = retrieve(message, agent=self.agent_id, verbose=verbose)
        context = format_context(chunks)
        sources = [
            f"{c.get('source_file')} p.{c.get('page_number')}"
            for c in chunks
        ]

        # 2. Build system prompt with RAG context
        system = self._build_system_prompt(context)

        # 3. Inject consistency block into message if present
        user_message = message
        if consistency_block:
            user_message = (
                f"{consistency_block}\n\n"
                f"---\n\n"
                f"{message}"
            )

        # 4. Build messages array
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

        raw_response = response.choices[0].message.content

        # 6. Separate <think> block from actual response (DeepSeek R1 style)
        thinking, answer = _parse_thinking(raw_response)

        return {
            "speaker":     self.agent_id,
            "name":        self.name,
            "response":    answer,
            "thinking":    thinking,
            "rag_sources": sources,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_thinking(text: str) -> tuple[str, str]:
    """
    Separate <think>...</think> block from the actual response.
    Returns (thinking, answer).
    """
    import re
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer   = text[think_match.end():].strip()
    else:
        thinking = ""
        answer   = text.strip()
    return thinking, answer


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    agent_id = sys.argv[1] if len(sys.argv) > 1 else "finance"
    message  = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else (
        "The Infrastructure CS is proposing a KSh 50 billion SGR extension "
        "to be financed through a new sovereign bond. What is your position?"
    )

    print(f"\nAgent  : {agent_id}")
    print(f"Message: {message}\n")

    agent  = Agent.from_config(agent_id)
    result = agent.speak(message, verbose=True)

    print(f"\n── {result['name']} ──\n")
    if result["thinking"]:
        print(f"[Thinking]\n{result['thinking'][:300]}...\n")
    print(f"[Response]\n{result['response']}")
    print(f"\n[Sources]\n" + "\n".join(result["rag_sources"]))