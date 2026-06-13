"""
debate_graph.py
───────────────
NetworkX knowledge graph layer for the Kenya AI Executive Roundtable.

Runs post-hoc after a debate transcript is complete. Extracts structured
claims from each agent turn via a single Groq call per turn, builds a
directed knowledge graph, analyses it for contested sources, contradictions,
coalitions, and open conflicts, then returns enriched insights for the
President's synthesis prompt.

Graph schema
────────────
Node types:
  agent     — one per participant  {agent_id, name, title}
  claim     — one per assertion    {text, turn, speaker, fiscal_year, figures}
  document  — one per source file  {source_file, domain, fiscal_year, priority}
  topic     — one per sub-topic    {label}

Edge types:
  (agent)    --MADE-->       (claim)    {turn}
  (claim)    --CITES-->      (document) {page}
  (agent)    --CHALLENGES--> (claim)    {turn, reason}
  (agent)    --SUPPORTS-->   (claim)    {turn}
  (claim)    --CONTRADICTS-> (claim)    {reason}    # same agent, different turns
  (claim)    --OPPOSES-->    (claim)    {reason}    # different agents
  (claim)    --ADDRESSES-->  (topic)    {}
  (document) --CITED_BY-->   (agent)    {turns: [int]}

Usage (standalone):
    from debate_graph import DebateGraph
    dg = DebateGraph(topic, transcript)
    insights = dg.analyse()
    print(dg.summary_report())

Usage (integrated into debate.py):
    from debate_graph import DebateGraph
    dg = DebateGraph(topic, transcript)
    enriched_prompt = dg.build_synthesis_context()
    # Pass enriched_prompt to president_synthesise()

Dependencies:
    pip install networkx groq python-dotenv
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_MODEL         = "llama-3.3-70b-versatile"
EXTRACT_MAX_TOKENS = 400
EXTRACT_TEMPERATURE = 0.0

AGENT_NAMES = {
    "finance":        "Prof. Kamau",
    "education":      "Dr. Njeri",
    "agriculture":    "Dr. Achieng",
    "ict":            "Eng. Mwangi",
    "infrastructure": "Eng. Otieno",
    "anticorruption": "Justice Waweru",
    "president":      "The President",
}

# ── Claim extraction prompt ───────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are extracting structured claims from a Kenya Cabinet debate turn.

Agent: {agent_name} ({agent_id})
Turn: {turn}
Debate topic: {topic}

Agent's response:
{response}

Sources cited (from RAG retrieval):
{sources}

Extract 2-4 factual claims from this response. For each claim:
- text: the specific claim as a concise sentence (include figures where present)
- figures: list of specific numbers/percentages/KSh amounts mentioned (empty list if none)
- fiscal_year: fiscal year this claim refers to (e.g. "2022_23") or "na" if timeless
- opposes_claim_about: what position this claim opposes (empty string if not adversarial)
- supports_claim_about: what position this claim supports (empty string if not supportive)
- sub_topics: list of 1-2 sub-topics this claim addresses from:
  [fiscal_policy, public_debt, revenue, monetary_policy, food_security, education_funding,
   infrastructure_cost, digital_economy, governance, procurement, devolution, sgr,
   energy, roads, agriculture, health, climate, employment]

Return ONLY valid JSON — no preamble, no markdown:
{{"claims": [
  {{"text": "...", "figures": ["..."], "fiscal_year": "...",
    "opposes_claim_about": "...", "supports_claim_about": "...",
    "sub_topics": ["..."]}}
]}}"""

# ── Groq client ───────────────────────────────────────────────────────────────

_groq_client: Optional[Groq] = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _groq_call_with_retry(fn, retries: int = 3, delay: float = 5.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            err = str(e)
            if any(x in err for x in ["Connection reset", "Connection aborted",
                                       "timed out", "rate_limit", "503", "502"]):
                wait = delay * (attempt + 1)
                print(f"    [Groq transient error, retrying in {wait:.0f}s...]")
                time.sleep(wait)
                global _groq_client
                _groq_client = None
                continue
            raise
    raise last_exc


# ── Claim extractor ───────────────────────────────────────────────────────────

def extract_claims(
    turn: dict,
    topic: str,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Extract structured claims from one debate turn via Groq.

    Returns list of claim dicts. Returns [] on any failure (safe).
    """
    agent_id   = turn.get("speaker", "unknown")
    agent_name = AGENT_NAMES.get(agent_id, agent_id)
    response   = turn.get("response", "").strip()
    sources    = ", ".join(turn.get("rag_sources", [])) or "none"

    if not response or agent_id == "president":
        return []

    prompt = _EXTRACT_PROMPT.format(
        agent_name = agent_name,
        agent_id   = agent_id,
        turn       = turn.get("turn", "?"),
        topic      = topic,
        response   = response[:1500],
        sources    = sources,
    )

    try:
        def _call():
            return _get_groq().chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = EXTRACT_MAX_TOKENS,
                temperature = EXTRACT_TEMPERATURE,
            )
        resp = _groq_call_with_retry(_call)
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        claims = data.get("claims", [])
        time.sleep(sleep_between)
        return claims

    except Exception as e:
        print(f"    [Claim extraction failed for turn {turn.get('turn')}: {e}]")
        return []


# ── DebateGraph ───────────────────────────────────────────────────────────────

class DebateGraph:
    """
    Post-hoc knowledge graph for a completed Cabinet debate.

    Build:
        dg = DebateGraph(topic, transcript)
        dg.build()          # extracts claims, builds graph (Groq calls)

    Query:
        insights = dg.analyse()
        report   = dg.summary_report()
        context  = dg.build_synthesis_context()   # for President's prompt

    Save/load:
        dg.save("graph_data.json")
        dg2 = DebateGraph.load("graph_data.json")
    """

    def __init__(self, topic: str, transcript: list[dict]):
        self.topic      = topic
        self.transcript = transcript
        self.G          = nx.DiGraph()
        self._built     = False

        # Raw extracted claims per turn: {turn_num: [claim_dict]}
        self._turn_claims: dict[int, list[dict]] = {}

        # Document registry: {source_file: set of agents who cited it}
        self._doc_agents: dict[str, set] = defaultdict(set)

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, verbose: bool = True) -> "DebateGraph":
        """Extract claims from all turns and build the graph.

        Makes one Groq call per non-president turn.
        For a 20-turn debate: ~18 calls, ~5,000 tokens total.
        """
        if verbose:
            print("\n  [DebateGraph] Extracting claims from transcript...")

        # ── Step 1: seed agent nodes ───────────────────────────────────────────
        speakers = set(t["speaker"] for t in self.transcript)
        for agent_id in speakers:
            self.G.add_node(
                f"agent:{agent_id}",
                node_type  = "agent",
                agent_id   = agent_id,
                name       = AGENT_NAMES.get(agent_id, agent_id),
            )

        # ── Step 2: seed document nodes from RAG sources ───────────────────────
        for turn in self.transcript:
            agent_id = turn.get("speaker")
            for source_str in turn.get("rag_sources", []):
                # source_str format: "filename.pdf pN"
                parts     = source_str.rsplit(" p", 1)
                src_file  = parts[0].strip()
                page      = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                doc_node  = f"doc:{src_file}"

                if not self.G.has_node(doc_node):
                    self.G.add_node(
                        doc_node,
                        node_type   = "document",
                        source_file = src_file,
                        cited_by    = [],
                        pages       = [],
                    )

                # Track which agents cited which docs
                self._doc_agents[src_file].add(agent_id)
                node_data = self.G.nodes[doc_node]
                if agent_id not in node_data.get("cited_by", []):
                    node_data.setdefault("cited_by", []).append(agent_id)
                if page and page not in node_data.get("pages", []):
                    node_data.setdefault("pages", []).append(page)

                # Edge: agent CITED_BY doc (reversed for easy traversal)
                self.G.add_edge(
                    f"agent:{agent_id}", doc_node,
                    edge_type = "CITES_DOC",
                    turn      = turn.get("turn", 0),
                )

        # ── Step 3: extract claims per turn ────────────────────────────────────
        all_claims: list[tuple[int, str, dict]] = []  # (turn, agent_id, claim)

        for turn in self.transcript:
            agent_id = turn.get("speaker")
            turn_num = turn.get("turn", 0)

            if agent_id == "president":
                continue

            if verbose:
                print(f"    Turn {turn_num}: {AGENT_NAMES.get(agent_id, agent_id)}...")

            raw_claims = extract_claims(turn, self.topic)
            self._turn_claims[turn_num] = raw_claims

            for ci, claim in enumerate(raw_claims):
                claim_id   = f"claim:{agent_id}:t{turn_num}:c{ci}"
                claim_text = claim.get("text", "").strip()
                if not claim_text:
                    continue

                # Add claim node
                self.G.add_node(
                    claim_id,
                    node_type    = "claim",
                    text         = claim_text,
                    turn         = turn_num,
                    speaker      = agent_id,
                    figures      = claim.get("figures", []),
                    fiscal_year  = claim.get("fiscal_year", "na"),
                    sub_topics   = claim.get("sub_topics", []),
                    opposes      = claim.get("opposes_claim_about", ""),
                    supports     = claim.get("supports_claim_about", ""),
                )

                # Edge: agent MADE claim
                self.G.add_edge(
                    f"agent:{agent_id}", claim_id,
                    edge_type = "MADE",
                    turn      = turn_num,
                )

                # Edges: claim ADDRESSES sub-topics
                for sub_topic in claim.get("sub_topics", []):
                    topic_node = f"topic:{sub_topic}"
                    if not self.G.has_node(topic_node):
                        self.G.add_node(topic_node, node_type="topic", label=sub_topic)
                    self.G.add_edge(claim_id, topic_node, edge_type="ADDRESSES")

                all_claims.append((turn_num, agent_id, claim_id, claim))

        # ── Step 4: detect oppositions and supports ────────────────────────────
        # For each claim with opposes_claim_about, find the closest matching claim
        # from a different agent and add an OPPOSES edge.
        claim_nodes = [
            (n, d) for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "claim"
        ]

        for cid, cdata in claim_nodes:
            opposes_text = cdata.get("opposes", "").strip()
            supports_text = cdata.get("supports", "").strip()

            if opposes_text:
                # Find a claim from a DIFFERENT agent that this opposes
                target = self._find_matching_claim(
                    cid, opposes_text, same_agent=False
                )
                if target:
                    self.G.add_edge(
                        cid, target,
                        edge_type = "OPPOSES",
                        reason    = opposes_text[:100],
                    )

            if supports_text:
                # Find a claim from ANY agent that this supports
                target = self._find_matching_claim(
                    cid, supports_text, same_agent=None
                )
                if target:
                    self.G.add_edge(
                        cid, target,
                        edge_type = "SUPPORTS",
                        reason    = supports_text[:100],
                    )

        # ── Step 5: detect self-contradictions (same agent, opposing claims) ───
        agent_claims: dict[str, list[str]] = defaultdict(list)
        for cid, cdata in claim_nodes:
            agent_claims[cdata.get("speaker", "")].append(cid)

        for agent_id, cids in agent_claims.items():
            for i, cid_a in enumerate(cids):
                for cid_b in cids[i + 1:]:
                    text_a = self.G.nodes[cid_a].get("text", "").lower()
                    text_b = self.G.nodes[cid_b].get("text", "").lower()
                    # Simple heuristic: claims about same sub_topic but different turns
                    topics_a = set(self.G.nodes[cid_a].get("sub_topics", []))
                    topics_b = set(self.G.nodes[cid_b].get("sub_topics", []))
                    shared = topics_a & topics_b
                    # Check if one has figures and the other has different figures
                    figs_a = set(self.G.nodes[cid_a].get("figures", []))
                    figs_b = set(self.G.nodes[cid_b].get("figures", []))
                    if shared and figs_a and figs_b and not (figs_a & figs_b):
                        # Different figures on same sub_topic = potential contradiction
                        self.G.add_edge(
                            cid_a, cid_b,
                            edge_type = "POSSIBLE_CONTRADICTION",
                            agent     = agent_id,
                            topics    = list(shared),
                        )

        self._built = True
        if verbose:
            print(f"  [DebateGraph] Built: {self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")
        return self

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _find_matching_claim(
        self,
        source_claim_id: str,
        target_text: str,
        same_agent,   # True = same, False = different, None = any
    ) -> Optional[str]:
        """Find a claim node whose text best matches target_text."""
        source_agent = self.G.nodes[source_claim_id].get("speaker", "")
        target_lower = target_text.lower()
        best_id      = None
        best_overlap = 0

        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("node_type") != "claim":
                continue
            if nid == source_claim_id:
                continue

            node_agent = ndata.get("speaker", "")
            if same_agent is True and node_agent != source_agent:
                continue
            if same_agent is False and node_agent == source_agent:
                continue

            # Jaccard word overlap
            claim_lower = ndata.get("text", "").lower()
            words_target = set(target_lower.split())
            words_claim  = set(claim_lower.split())
            if not words_target or not words_claim:
                continue
            overlap = len(words_target & words_claim) / len(words_target | words_claim)
            if overlap > best_overlap and overlap > 0.12:
                best_overlap = overlap
                best_id      = nid

        return best_id

    # ── Analysis ───────────────────────────────────────────────────────────────

    def analyse(self) -> dict:
        """Run full graph analysis. Returns structured insights dict."""
        if not self._built:
            raise RuntimeError("Call .build() before .analyse()")

        return {
            "contested_documents":  self._contested_documents(),
            "self_contradictions":  self._self_contradictions(),
            "coalitions":           self._coalitions(),
            "open_conflicts":       self._open_conflicts(),
            "most_cited_document":  self._most_cited_document(),
            "dominant_sub_topics":  self._dominant_sub_topics(),
            "agent_claim_counts":   self._agent_claim_counts(),
            "cross_agent_oppositions": self._cross_agent_oppositions(),
        }

    def _contested_documents(self) -> list[dict]:
        """Documents cited by 2+ agents — potential shared battleground."""
        contested = []
        for src_file, agents in self._doc_agents.items():
            if len(agents) >= 2:
                contested.append({
                    "source_file": src_file,
                    "cited_by":    sorted(agents),
                    "agent_names": [AGENT_NAMES.get(a, a) for a in sorted(agents)],
                    "num_agents":  len(agents),
                })
        return sorted(contested, key=lambda x: x["num_agents"], reverse=True)

    def _self_contradictions(self) -> list[dict]:
        """Claims from the same agent that contradict each other."""
        contras = []
        for u, v, edata in self.G.edges(data=True):
            if edata.get("edge_type") == "POSSIBLE_CONTRADICTION":
                agent_id   = edata.get("agent", "")
                claim_a    = self.G.nodes[u].get("text", "")
                claim_b    = self.G.nodes[v].get("text", "")
                turn_a     = self.G.nodes[u].get("turn", 0)
                turn_b     = self.G.nodes[v].get("turn", 0)
                contras.append({
                    "agent":       agent_id,
                    "agent_name":  AGENT_NAMES.get(agent_id, agent_id),
                    "claim_a":     claim_a[:120],
                    "turn_a":      turn_a,
                    "claim_b":     claim_b[:120],
                    "turn_b":      turn_b,
                    "sub_topics":  edata.get("topics", []),
                })
        return contras

    def _coalitions(self) -> list[dict]:
        """Pairs of agents who mutually support each other's claims."""
        support_pairs: dict[frozenset, int] = defaultdict(int)

        for u, v, edata in self.G.edges(data=True):
            if edata.get("edge_type") == "SUPPORTS":
                agent_u = self.G.nodes[u].get("speaker", "")
                agent_v = self.G.nodes[v].get("speaker", "")
                if agent_u and agent_v and agent_u != agent_v:
                    support_pairs[frozenset([agent_u, agent_v])] += 1

        coalitions = []
        for pair, count in sorted(support_pairs.items(), key=lambda x: x[1], reverse=True):
            agents = sorted(pair)
            coalitions.append({
                "agents":       agents,
                "agent_names":  [AGENT_NAMES.get(a, a) for a in agents],
                "support_count": count,
            })
        return coalitions

    def _open_conflicts(self) -> list[dict]:
        """OPPOSES edges with no corresponding SUPPORTS — unresolved tensions."""
        conflicts = []
        for u, v, edata in self.G.edges(data=True):
            if edata.get("edge_type") == "OPPOSES":
                # Check if anyone supports the target claim
                supported = any(
                    d.get("edge_type") == "SUPPORTS"
                    for _, t, d in self.G.out_edges(u, data=True)
                    if t == v
                )
                if not supported:
                    agent_u = self.G.nodes[u].get("speaker", "")
                    agent_v = self.G.nodes[v].get("speaker", "")
                    conflicts.append({
                        "opposing_agent":  agent_u,
                        "opposing_name":   AGENT_NAMES.get(agent_u, agent_u),
                        "target_agent":    agent_v,
                        "target_name":     AGENT_NAMES.get(agent_v, agent_v),
                        "claim":           self.G.nodes[u].get("text", "")[:120],
                        "opposes":         self.G.nodes[v].get("text", "")[:120],
                        "reason":          edata.get("reason", "")[:100],
                    })
        return conflicts

    def _most_cited_document(self) -> Optional[dict]:
        """The document cited by the most agents."""
        if not self._doc_agents:
            return None
        src = max(self._doc_agents, key=lambda k: len(self._doc_agents[k]))
        return {
            "source_file": src,
            "cited_by":    sorted(self._doc_agents[src]),
            "agent_names": [AGENT_NAMES.get(a, a) for a in sorted(self._doc_agents[src])],
        }

    def _dominant_sub_topics(self) -> list[dict]:
        """Sub-topics by number of claims that address them."""
        topic_counts: dict[str, int] = defaultdict(int)
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") == "topic":
                # Count in-edges (claims addressing this topic)
                topic_counts[d["label"]] = self.G.in_degree(n)
        return [
            {"sub_topic": t, "claim_count": c}
            for t, c in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            if c > 0
        ]

    def _agent_claim_counts(self) -> dict[str, int]:
        """Number of claims extracted per agent."""
        counts: dict[str, int] = defaultdict(int)
        for _, d in self.G.nodes(data=True):
            if d.get("node_type") == "claim":
                counts[d.get("speaker", "unknown")] += 1
        return dict(counts)

    def _cross_agent_oppositions(self) -> list[dict]:
        """All OPPOSES edges between different agents."""
        oppositions = []
        for u, v, edata in self.G.edges(data=True):
            if edata.get("edge_type") == "OPPOSES":
                agent_u = self.G.nodes[u].get("speaker", "")
                agent_v = self.G.nodes[v].get("speaker", "")
                if agent_u != agent_v:
                    oppositions.append({
                        "from_agent":  agent_u,
                        "from_name":   AGENT_NAMES.get(agent_u, agent_u),
                        "to_agent":    agent_v,
                        "to_name":     AGENT_NAMES.get(agent_v, agent_v),
                        "claim":       self.G.nodes[u].get("text", "")[:120],
                        "opposes":     self.G.nodes[v].get("text", "")[:120],
                    })
        return oppositions

    # ── Summary report ─────────────────────────────────────────────────────────

    def summary_report(self) -> str:
        """Human-readable summary of graph analysis."""
        insights = self.analyse()
        lines    = []

        lines.append("═" * 60)
        lines.append("  DEBATE KNOWLEDGE GRAPH — ANALYSIS REPORT")
        lines.append(f"  Topic: {self.topic}")
        lines.append("═" * 60)

        # Graph stats
        n_claims = sum(
            1 for _, d in self.G.nodes(data=True)
            if d.get("node_type") == "claim"
        )
        n_docs = sum(
            1 for _, d in self.G.nodes(data=True)
            if d.get("node_type") == "document"
        )
        lines.append(f"\n  Nodes  : {self.G.number_of_nodes()} "
                     f"({n_claims} claims, {n_docs} documents)")
        lines.append(f"  Edges  : {self.G.number_of_edges()}")

        # Claims per agent
        lines.append("\n  Claims per agent:")
        for agent_id, count in sorted(
            insights["agent_claim_counts"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"    {AGENT_NAMES.get(agent_id, agent_id):<25} {count} claims")

        # Dominant sub-topics
        if insights["dominant_sub_topics"]:
            lines.append("\n  Dominant sub-topics:")
            for item in insights["dominant_sub_topics"][:5]:
                lines.append(f"    {item['sub_topic']:<30} {item['claim_count']} claims")

        # Contested documents
        if insights["contested_documents"]:
            lines.append("\n  Contested documents (cited by multiple agents):")
            for doc in insights["contested_documents"][:5]:
                lines.append(
                    f"    {doc['source_file'][:50]}\n"
                    f"      ↳ {', '.join(doc['agent_names'])}"
                )

        # Coalitions
        if insights["coalitions"]:
            lines.append("\n  Coalitions (mutual support):")
            for c in insights["coalitions"][:3]:
                lines.append(
                    f"    {' + '.join(c['agent_names'])} "
                    f"({c['support_count']} mutual supports)"
                )

        # Open conflicts
        if insights["open_conflicts"]:
            lines.append("\n  Unresolved conflicts:")
            for cf in insights["open_conflicts"][:4]:
                lines.append(
                    f"    {cf['opposing_name']} ↔ {cf['target_name']}\n"
                    f"      \"{cf['claim'][:80]}...\""
                )

        # Self-contradictions
        if insights["self_contradictions"]:
            lines.append("\n  Possible self-contradictions:")
            for sc in insights["self_contradictions"][:3]:
                lines.append(
                    f"    {sc['agent_name']} (turns {sc['turn_a']} vs {sc['turn_b']})\n"
                    f"      Turn {sc['turn_a']}: \"{sc['claim_a'][:70]}...\"\n"
                    f"      Turn {sc['turn_b']}: \"{sc['claim_b'][:70]}...\""
                )

        lines.append("\n" + "═" * 60)
        return "\n".join(lines)

    # ── Synthesis context builder ──────────────────────────────────────────────

    def build_synthesis_context(self) -> str:
        """Build the graph-enriched context block for the President's synthesis.

        This is injected into the synthesis prompt so the President's final
        policy directive is grounded in the structural patterns of the debate,
        not just the last few turns.
        """
        insights = self.analyse()

        parts = []
        parts.append("KNOWLEDGE GRAPH ANALYSIS OF THIS DEBATE:\n")

        # Most cited document — the anchor
        if insights["most_cited_document"]:
            doc = insights["most_cited_document"]
            parts.append(
                f"ANCHOR DOCUMENT: '{doc['source_file']}' was cited by "
                f"{', '.join(doc['agent_names'])} — the factual anchor of this debate.\n"
            )

        # Contested documents — same source, potentially different readings
        contested = insights["contested_documents"]
        if contested:
            parts.append("CONTESTED SOURCES (same document, multiple agents):")
            for doc in contested[:3]:
                parts.append(
                    f"  • '{doc['source_file']}' cited by: "
                    f"{', '.join(doc['agent_names'])}"
                )
                parts.append(
                    "    → These agents may be drawing different conclusions "
                    "from the same data. Verify consistency."
                )
            parts.append("")

        # Coalitions — who is aligned
        coalitions = insights["coalitions"]
        if coalitions:
            parts.append("COALITIONS DETECTED (agents with aligned positions):")
            for c in coalitions[:3]:
                parts.append(
                    f"  • {' and '.join(c['agent_names'])} "
                    f"— {c['support_count']} instances of mutual support"
                )
            parts.append("")

        # Open conflicts — unresolved tensions
        conflicts = insights["open_conflicts"]
        if conflicts:
            parts.append("UNRESOLVED CONFLICTS (require Presidential resolution):")
            for cf in conflicts[:4]:
                parts.append(
                    f"  • {cf['opposing_name']} vs {cf['target_name']}:\n"
                    f"    {cf['opposing_name']} argues: \"{cf['claim'][:100]}\"\n"
                    f"    This opposes {cf['target_name']}'s position on: \"{cf['opposes'][:100]}\""
                )
            parts.append("")

        # Self-contradictions — internal consistency flags
        contras = insights["self_contradictions"]
        if contras:
            parts.append("INTERNAL CONSISTENCY FLAGS (same agent, conflicting claims):")
            for sc in contras[:3]:
                parts.append(
                    f"  • {sc['agent_name']} may have shifted position on "
                    f"{', '.join(sc['sub_topics'])}:\n"
                    f"    Turn {sc['turn_a']}: \"{sc['claim_a'][:80]}\"\n"
                    f"    Turn {sc['turn_b']}: \"{sc['claim_b'][:80]}\""
                )
            parts.append("")

        # Dominant sub-topics — what the debate was really about
        sub_topics = insights["dominant_sub_topics"]
        if sub_topics:
            topic_str = ", ".join(
                f"{t['sub_topic']} ({t['claim_count']} claims)"
                for t in sub_topics[:5]
            )
            parts.append(f"CORE DEBATE THEMES: {topic_str}\n")

        parts.append(
            "USE THE ABOVE to:\n"
            "1. Resolve unresolved conflicts with a clear Presidential directive.\n"
            "2. Note where different ministers cited the same data differently.\n"
            "3. Identify which coalition position (if any) has the stronger evidence base.\n"
            "4. Flag any agent who appeared to contradict themselves.\n"
            "5. Deliver a final, specific policy decision — not a summary."
        )

        return "\n".join(parts)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save graph data and extracted claims to JSON for inspection/reload."""
        data = {
            "topic":       self.topic,
            "nodes":       [
                {"id": n, **d}
                for n, d in self.G.nodes(data=True)
            ],
            "edges":       [
                {"source": u, "target": v, **d}
                for u, v, d in self.G.edges(data=True)
            ],
            "turn_claims": {
                str(k): v for k, v in self._turn_claims.items()
            },
            "doc_agents":  {
                k: list(v) for k, v in self._doc_agents.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [DebateGraph] Saved → {path}")

    @classmethod
    def from_saved(cls, path: str) -> "DebateGraph":
        """Reload a saved graph without re-running Groq extraction."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        dg = cls(topic=data["topic"], transcript=[])
        G  = nx.DiGraph()

        for node in data["nodes"]:
            nid = node.pop("id")
            G.add_node(nid, **node)

        for edge in data["edges"]:
            src = edge.pop("source")
            tgt = edge.pop("target")
            G.add_edge(src, tgt, **edge)

        dg.G           = G
        dg._turn_claims = {int(k): v for k, v in data["turn_claims"].items()}
        dg._doc_agents  = defaultdict(set, {
            k: set(v) for k, v in data["doc_agents"].items()
        })
        dg._built = True
        return dg


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and analyse a debate knowledge graph"
    )
    parser.add_argument(
        "--transcript", "-t", required=True,
        help="Path to debate transcript JSON (from debate.py --save)"
    )
    parser.add_argument(
        "--save-graph", "-g", default=None,
        help="Save graph data to this JSON file"
    )
    parser.add_argument(
        "--load-graph", "-l", default=None,
        help="Load pre-built graph from JSON (skips Groq extraction)"
    )
    args = parser.parse_args()

    # Load transcript
    with open(args.transcript, encoding="utf-8") as f:
        saved = json.load(f)

    topic      = saved["topic"]
    transcript = saved["transcript"]

    if args.load_graph:
        print(f"Loading pre-built graph from {args.load_graph}...")
        dg = DebateGraph.from_saved(args.load_graph)
    else:
        dg = DebateGraph(topic, transcript)
        dg.build(verbose=True)
        if args.save_graph:
            dg.save(args.save_graph)

    print(dg.summary_report())
    print("\n" + "═" * 60)
    print("SYNTHESIS CONTEXT (would be injected into President's prompt):")
    print("═" * 60)
    print(dg.build_synthesis_context())