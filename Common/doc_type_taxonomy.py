"""
doc_type_taxonomy.py
────────────────────
Single source of truth for the Kenya AI Executive Roundtable RAG pipeline.

Contains:
  - Document type detection rules (filename + cover text)
  - Agent assignment rules (primary_agents, issuing_agent, agent_access)
  - Topic taxonomy (SUBJECT_TOPICS + AGENDA_FRAMES)
  - Coalition configuration (per-topic agent caps)
  - Domain taxonomy (per doc_type)
  - Chunking strategy maps
  - Priority and RAG weight maps
  - Manual overrides

Import this module in tag_documents.py, chunk_documents.py, and retriever.py.
Nothing else should hardcode any of these values.

FIXES APPLIED (v2):
  FIX 1 — match_primary_agents: warns on >2 agent collisions, logs to unknown_doc_types.log
  FIX 2 — build_agent_access: None sentinel replaces empty-list "primary+president" convention;
           ADD_PRESIDENT_TO_ALL set makes the rule explicit and readable
  FIX 3 — extract_doc_year: prefers the later year in YYYY-YYYY / YYYY-YY range filenames
  FIX 4 — SKIP_SECTIONS_MAP: explicit entries for all previously unmapped doc_types
  FIX 5 — COALITION_CONFIG: added monetary_policy coalition entry
  FIX 6 — MANUAL_OVERRIDES: added TODO comment block for BPS superseded series
"""

import re
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

ALL_AGENTS = [
    "finance",
    "education",
    "agriculture",
    "ict",
    "infrastructure",
    "anticorruption",
    "president",
]

AGENT_DESCRIPTIONS = {
    "finance":        "CS Finance & National Treasury — fiscal policy, public debt, tax, CBK, KRA, budget",
    "education":      "CS Education — KNEC, KICD, TSC, curriculum, TVET, universities, school funding",
    "agriculture":    "CS Agriculture — KALRO, AFA, KEPHIS, food security, irrigation, livestock, crops",
    "ict":            "CS ICT & Digital Economy — CA, KICTANET, digital policy, cybersecurity, fintech, Konza",
    "infrastructure": "CS Infrastructure — KeNHA, KURA, KeRRA, KETRACO, KPA, KenGen, roads, energy, ports, SGR",
    "anticorruption": "CS Anti-Corruption — EACC, PPRA, PPOA, ODPP, audit, procurement, governance",
    "president":      "The President — cross-cutting national policy, Vision 2030, BETA agenda, devolution",
}


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT TYPE DETECTION — FILENAME RULES
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENT_TYPE_RULES = [
    # ── Finance: pure data / tables ───────────────────────────────────────────
    (r"pure.?table",                                         "pure_tables"),
    (r"statistical.?annex",                                  "statistical_annex"),
    (r"estimates.?revenue.?grants|revenue.?grants.?loans",   "revenue_grants_estimates"),
    (r"annex.?state.?corp",                                  "state_corporations_annex"),

    # ── Finance: fiscal policy ────────────────────────────────────────────────
    (r"budget.policy.statement|^bps",                        "budget_policy_statement"),
    (r"budget.*review.*outlook|brop",                        "budget_review_outlook"),
    (r"budget.?summary",                                     "budget_summary"),
    (r"budget.?speech",                                      "budget_speech"),
    (r"post.?election.?economic",                            "post_election_report"),
    (r"medium.?term.?debt|mtds",                             "debt_management_strategy"),
    (r"annual.?public.?debt|public.?debt.?report",           "public_debt_report"),

    # ── Finance: controller of budget ─────────────────────────────────────────
    (r"ngbirr",                                              "controller_of_budget"),
    (r"annual.?birr|ng.?annual.?birr",                       "controller_of_budget"),
    (r"national.?government.?budget.?impl",                  "controller_of_budget"),
    (r"controller.?of.?budget",                              "controller_of_budget"),
    (r"national.?government.?oct",                           "controller_of_budget"),
    (r"national.?book",                                      "controller_of_budget"),
    (r"national.?budget.?may",                               "controller_of_budget"),
    (r"national.?report.?book",                              "controller_of_budget"),
    (r"september.?book",                                     "controller_of_budget"),
    (r"national.?government.?book",                          "controller_of_budget"),
    (r"annual.?national.?government.?budget",                "controller_of_budget"),

    # ── Finance: CBK ──────────────────────────────────────────────────────────
    (r"cbk.*annual.*report|cbk_\d{4}|cnk.*annual",          "cbk_annual_report"),
    (r"cbk.*mpc|monetary.?policy.?comm",                     "cbk_mpc_report"),
    (r"cbk.*fsr|financial.?sector.?stab",                    "cbk_fsr_report"),
    (r"monetary.?policy.?statement",                         "cbk_mpc_report"),

    # ── Finance: KRA ──────────────────────────────────────────────────────────
    (r"annual.?revenue.?performance|revenue.?performance",   "kra_revenue_performance"),
    (r"annual.?corporate.?report|corporate.?report",         "kra_corporate_plan"),
    (r"kra.*corporate|corporate.*plan.*kra",                 "kra_corporate_plan"),
    (r"tax.?expenditure",                                    "tax_expenditure_report"),
    (r"kra.*plan|corporate.*plan",                           "kra_corporate_plan"),

    # ── Finance: macroeconomic ────────────────────────────────────────────────
    (r"economic.?survey",                                    "economic_survey"),
    (r"imf",                                                 "imf_report"),
    (r"world.?bank|kenya.?economic.?update",                 "world_bank_report"),

    # ── Finance: legal ────────────────────────────────────────────────────────
    (r"finance.?act",                                        "finance_act"),
    (r"finance.?bill",                                       "finance_bill"),
    (r"constitution",                                        "constitution"),

    # ── Finance: auditor general ──────────────────────────────────────────────
    (r"auditor.?general|auditor.?gen",                       "auditor_general_report"),
    (r"summary.?report.?auditor",                            "auditor_general_report"),
    (r"national.?government.?audit",                         "auditor_general_report"),
    (r"report.?auditor.?general",                            "auditor_general_report"),
    (r"summary.?report.*national.*government",               "auditor_general_report"),

    # ── Finance: forensic audit ───────────────────────────────────────────────
    (r"forensic.?audit",                                     "forensic_audit"),

    # ── AntiCorruption ────────────────────────────────────────────────────────
    (r"eacc|ethics.anti.corruption",                         "eacc_report"),
    (r"ppra|ppoa",                                           "ppra_report"),
    (r"odpp|director.of.public.prosecution",                 "odpp_report"),
    (r"bribery.?act",                                        "guidelines"),
    (r"anticorruption|anti.corruption.*act",                 "act"),
    (r"public.?officer.?ethics",                             "act"),
    (r"debarment",                                           "manual"),
    (r"fatf.?recommendations",                               "guidelines"),
    (r"mer.?of.?kenya|mutual.?evaluation",                   "mer_report"),
    (r"prosecution.*cec|cec.*prosecution",                   "guidelines"),
    (r"amendments.*regulations|regulations.*amendments",     "regulations"),

    # ── Education ─────────────────────────────────────────────────────────────
    (r"knec.*annual|annual.*knec",                           "annual_report"),
    (r"kcse.*statistics|kcse.*essential",                    "statistics_report"),
    (r"kjsea",                                               "statistics_report"),
    (r"basic.?education.*curriculum|curriculum.*framework",  "framework"),
    (r"basic.?education.*act",                               "act"),
    (r"national.*curriculum.*policy",                        "policy"),
    (r"education.*sector.*report|education.*report",         "sector_report"),
    (r"education.*strategic.*plan|education.*strategy",      "strategic_plan"),
    (r"kesp",                                                "strategic_plan"),
    (r"taskforce.*report",                                   "research_report"),
    (r"nasmla",                                              "research_report"),
    (r"gpe.*results|global.*partnership.*education",         "research_report"),
    (r"ngcdf.*audit|audit.*ngcdf",                           "audit_report"),
    (r"ngcdf",                                               "guidelines"),
    (r"capitation.*grant|infrastructure.*grant.*school",     "audit_report"),
    (r"special.*audit.*school|school.*audit",                "audit_report"),

    # ── Agriculture ───────────────────────────────────────────────────────────
    (r"kalro.*annual|annual.*kalro|kalro.*report",           "annual_report"),
    (r"dri.*annual|annual.*dri",                             "annual_report"),
    (r"bri.*annual|annual.*bri",                             "annual_report"),
    (r"hri.*annual|annual.*hri",                             "annual_report"),
    (r"gerri.*annual|annual.*gerri",                         "annual_report"),
    (r"fcri.*annual|annual.*fcri",                           "annual_report"),
    (r"sri.*content.*kalro",                                 "annual_report"),
    (r"kephis.*annual",                                      "annual_report"),
    (r"nia.*strategic|national.*irrigation.*strategic",      "strategic_plan"),
    (r"moalf.*strategic|strategic.*moalf",                   "strategic_plan"),
    (r"ministry.*strategic.*agriculture|agriculture.*strategic", "strategic_plan"),
    (r"kasep.*policy|agriculture.*policy",                   "policy"),
    (r"agriculture.*insurance.*policy",                      "policy"),
    (r"national.*agriculture.*research.*policy",             "policy"),
    (r"pyrethrum.*bill|repeal.*bill",                        "bill"),
    (r"fsrp",                                                "framework"),
    (r"esmf",                                                "framework"),
    (r"capacity.*building.*strategy.*agriculture",           "strategic_plan"),
    (r"field.?guide",                                        "guidelines"),
    (r"crop.*conditions.*bulletin",                          "bulletin"),
    (r"dew.*bulletin|drought.*early.*warning",               "bulletin"),
    (r"weather.*forecast|seasonal.*forecast",                "bulletin"),
    (r"food.*loss.*waste|post.*harvest",                     "policy"),
    (r"seeds.*regulations|fertilizer.*regulations",          "regulations"),
    (r"propagating.*seeds",                                  "regulations"),
    (r"agriculture.*production.*report",                     "statistics_report"),
    (r"census.*agriculture",                                 "survey"),
    (r"regulatory.*impact.*assessment.*agriculture",         "assessment"),
    (r"eu.*deforestation",                                   "regulations"),
    (r"livestock.*agenda",                                   "policy"),
    (r"avocado.*nairobi|ke2025",                             "research_report"),
    (r"grain.*feed.*annual",                                 "research_report"),
    (r"agricultural.*biotechnology",                         "research_report"),
    (r"roots.*tuber.*strategy",                              "strategic_plan"),
    (r"dvs.*magazine|magazine.*dvs",                         "magazine"),
    (r"dairy.*interventions|daima",                          "framework"),
    (r"water.*harvesting|in.situ.*water",                    "guidelines"),
    (r"catalogue.*breeds",                                   "catalogue"),
    (r"ncpb.*national.*food|national.*food.*reserve",        "regulations"),
    (r"national.*cassava|cassava.*conference",               "conference_report"),
    (r"nsafs.*facilitators|facilitators.*manual",            "manual"),
    (r"astgs",                                               "strategic_plan"),
    (r"public.*consultation.*seeds",                         "policy"),
    (r"migratory.*pests|invasive.*pests",                    "strategic_plan"),
    (r"consolidated.*summary.*ncce",                         "conference_report"),
    (r"land.*use.*policy",                                   "policy"),

    # ── ICT ───────────────────────────────────────────────────────────────────
    (r"kictanet.*annual",                                    "annual_report"),
    (r"kictanet.*spotlight|kictanet.*quarterly",             "bulletin"),
    (r"kenya.*igf|igf.*kenya|iggf",                          "igf_report"),
    (r"eaigf",                                               "igf_report"),
    (r"kigf",                                                "igf_report"),
    (r"africa.*tech.*policy.*summit",                        "conference_report"),
    (r"ca.*annual|communications.*authority.*annual",        "annual_report"),
    (r"konza.*strategic|konza.*abridged",                    "strategic_plan"),
    (r"konza.*financial|konza.*audited",                     "financial_statements"),
    (r"kenya.*ai.*strategy",                                 "strategic_plan"),
    (r"digital.*masterplan|ict.*masterplan",                 "masterplan"),
    (r"national.*ict.*policy|ict.*policy.*guidelines",       "policy"),
    (r"national.*broadband",                                 "strategic_plan"),
    (r"e.commerce.*strategy|ecommerce.*strategy",            "strategic_plan"),
    (r"national.*addressing.*bill",                          "bill"),
    (r"national.*addressing.*policy",                        "policy"),
    (r"computer.*misuse|cybercrime",                         "act"),
    (r"data.*protection.*act",                               "act"),
    (r"cybersecurity.*strategy",                             "strategic_plan"),
    (r"mobile.*payments",                                    "bill"),
    (r"ajira.*digital",                                      "research_report"),
    (r"huduma.*namba",                                       "research_report"),
    (r"finaccess|fin.*access.*household",                    "survey"),
    (r"county.*ict.*survey",                                 "survey"),
    (r"digital.*readiness",                                  "survey"),
    (r"digital.*economy.*kenya|kenya.*digital.*economy",     "research_report"),
    (r"digital.*skills.*africa",                             "research_report"),
    (r"digital.*health.*who|global.*digital.*health",        "strategic_plan"),
    (r"smart.*cities.*sustainable",                          "research_report"),
    (r"safer.*web.*women",                                   "research_report"),
    (r"advocacy.*digital.*rights|digital.*rights.*upr",      "research_report"),
    (r"global.*geopolitics.*digital",                        "conference_report"),
    (r"audience.*measurement.*industry",                     "statistics_report"),
    (r"optic.*fibre.*forensic|forensic.*optic.*fibre",       "forensic_audit"),
    (r"invest.*kenya.*bpo",                                  "research_report"),
    (r"ncsc.*annual",                                        "annual_report"),
    (r"kictanet.*election.*observer",                        "research_report"),
    (r"impact.*implementing.*huduma",                        "research_report"),
    (r"state.*ict.*report",                                  "statistics_report"),
    (r"kictanet.*10.*years",                                 "annual_report"),

    # ── Infrastructure ────────────────────────────────────────────────────────
    (r"kenha.*annual|annual.*kenha|kenya.*national.*highways.*annual", "annual_report"),
    (r"kura.*annual|annual.*kura",                           "annual_report"),
    (r"kerra.*annual|annual.*kerra|kerra.*fy",               "annual_report"),
    (r"ketraco.*annual|annual.*ketraco",                     "annual_report"),
    (r"kengen.*integrated|kengen.*annual",                   "annual_report"),
    (r"kpa.*financial|kenya.*ports.*annual",                 "annual_report"),
    (r"krc.*railways.*bill|railways.*bill",                  "bill"),
    (r"kura.*strategic|kura.*revised",                       "strategic_plan"),
    (r"kerra.*strategic|kerra.*final.*strategic",            "strategic_plan"),
    (r"kr.*strategic.*plan|kenya.*railways.*strategic",      "strategic_plan"),
    (r"kenya.*power.*sustainability",                        "strategic_plan"),
    (r"rdm\.\d+\.\d+|road.*design.*manual",                  "manual"),
    (r"road.*safety.*audit|pam.*road.*safety",               "guidelines"),
    (r"standard.*specification.*road.*bridge",               "guidelines"),
    (r"intelligent.*transport",                              "research_report"),
    (r"bus.*rapid.*transport",                               "research_report"),
    (r"epra.*annual|energy.*petroleum.*statistics",          "statistics_report"),
    (r"epra.*biannual|biannual.*energy",                     "statistics_report"),
    (r"energy.*petroleum.*report|petroleum.*report",         "statistics_report"),
    (r"lpg.*depot.*safety",                                  "audit_report"),
    (r"iea.*kenya",                                          "research_report"),
    (r"kengen.*carbon|carbon.*credits",                      "research_report"),
    (r"sep.*green.*mpa|lmp.*green.*mpa",                     "framework"),
    (r"esmf.*green|green.*esmf",                             "framework"),
    (r"esia.*addendum|statcom.*esia",                        "assessment"),
    (r"kosap.*success|success.*kosap",                       "research_report"),
    (r"kenya.*gdp.*quarter|quarterly.*gdp",                  "statistics_report"),
    (r"kenya.*leading.*economic.*indicators",                "statistics_report"),
    (r"regulatory.*impact.*assessment.*roads|roads.*regulatory", "assessment"),
    (r"kenya.*roads.*regulations|roads.*regulations",        "regulations"),
    (r"annual.*procurement.*plan|maintenance.*capp",         "guidelines"),
    (r"quality.*policy.*statement",                          "policy"),
    (r"service.*charter.*kerra|kerra.*charter",              "guidelines"),
    (r"alternative.*business.*partnerships",                 "research_report"),
    (r"energy.*post|the.*energy.*post",                      "bulletin"),
    (r"kplc|kenya.*power",                                   "annual_report"),
    (r"financial.*statement.*\d{4}",                         "financial_statements"),
    (r"upgrading.*nanyuki|strengthening.*kandwia",           "research_report"),

    # ── President / cross-cutting ─────────────────────────────────────────────
    (r"national.*values.*principles.*governance",            "guidelines"),
    (r"unodc.*odpp",                                         "guidelines"),
    (r"election.*observer",                                  "research_report"),
    (r"national.*land.*use.*policy",                         "policy"),

    # ── Catch-all ─────────────────────────────────────────────────────────────
    (r"strategic.*plan|strategic.?plan",                     "strategic_plan"),
    (r"annual.*report|report.*annual",                       "annual_report"),
    (r"audit.*report|report.*audit",                         "audit_report"),
    (r"financial.*statements|audited.*financial",            "financial_statements"),
    (r"guidelines|manual",                                   "guidelines"),
    (r"policy",                                              "policy"),
    (r"framework",                                           "framework"),
    (r"assessment",                                          "assessment"),
    (r"survey",                                              "survey"),
    (r"bulletin",                                            "bulletin"),
    (r"regulations",                                         "regulations"),
    (r"act\b",                                               "act"),
    (r"bill\b",                                              "bill"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT TYPE DETECTION — COVER TEXT RULES
# ══════════════════════════════════════════════════════════════════════════════

COVER_TYPE_RULES = [
    (r"budget.*policy.*statement",                           "budget_policy_statement"),
    (r"budget.*review.*outlook|budget.*outlook.*paper",      "budget_review_outlook"),
    (r"controller of budget",                                "controller_of_budget"),
    (r"budget implementation review",                        "controller_of_budget"),
    (r"annual corporate report",                             "kra_corporate_plan"),
    (r"kenya revenue authority.*annual",                     "kra_corporate_plan"),
    (r"auditor.general.*national government",                "auditor_general_report"),
    (r"report of the auditor",                               "auditor_general_report"),
    (r"monetary policy committee",                           "cbk_mpc_report"),
    (r"financial sector.*stability",                         "cbk_fsr_report"),
    (r"central bank of kenya.*annual report",                "cbk_annual_report"),
    (r"medium.term debt management",                         "debt_management_strategy"),
    (r"public debt.*management.*report",                     "public_debt_report"),
    (r"economic survey",                                     "economic_survey"),
    (r"international monetary fund|imf.*kenya",              "imf_report"),
    (r"world bank.*kenya|kenya.*economic update",            "world_bank_report"),
    (r"revenue.*performance.*report",                        "kra_revenue_performance"),
    (r"tax expenditure report",                              "tax_expenditure_report"),
    (r"kenya.*igf|internet.*governance.*forum",              "igf_report"),
    (r"kictanet",                                            "annual_report"),
    (r"konza.*technopolis",                                  "annual_report"),
    (r"energy.*petroleum.*statistics",                       "statistics_report"),
    (r"strategic plan",                                      "strategic_plan"),
    (r"annual report",                                       "annual_report"),
]


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY AGENT RULES
# ══════════════════════════════════════════════════════════════════════════════

AGENT_PATTERNS = {
    "finance": [
        r"cbk_|monetary.policy|mpc.report|cbk.*fsr|cbk.*annual",
        r"kra.*corporate|kra.*plan|revenue.performance",
        r"tax.expenditure",
        r"medium.term.debt|debt.management.strategy",
        r"public.debt.report|annual.public.debt",
        r"budget.policy.statement|budget.review.outlook|budget.summary",
        r"controller.of.budget|ngbirr|national.government.budget",
        r"estimates.of.revenue.grants|statistical.annex|annex.state.corp",
        r"pure.tables|revenue.grants",
        r"post.election.economic",
        r"finance.act|finance.bill",
        r"imf|world.bank|kenya.economic.update",
        r"economic.survey",
        r"auditor.generals.summary|national.government.audit|summary.report.*auditor",
        r"financial.statement.*\d{4}.*kpa|kpa.*financial",
        r"ngbirr|annual.birr",
        r"constitution",
    ],
    "education": [
        r"knec|kcse|kjsea",
        r"kicd|curriculum",
        r"basic.education",
        r"education.sector|ministry.*education",
        r"kesp|education.*strategic",
        r"taskforce.*report.*education|education.*taskforce",
        r"nasmla",
        r"gpe.*results",
        r"ngcdf",
        r"capitation.*grant|school.*audit|special.*audit.*school",
        r"teacher.*service|tsc",
        r"tvet",
        r"youth.in.agribusiness",
        r"digital.skills.*africa",
    ],
    "agriculture": [
        r"kalro|kephis|afa\.bulletin|afa.*second|afa.*first",
        r"field.guide|pastoral|agro.pastoral",
        r"crop.conditions|food.loss|post.harvest",
        r"agricultural.insurance|agricultural.biotechnology",
        r"roots.and.tuber|seeds|fertilizer|propagating",
        r"livestock.agenda|animal.disease",
        r"agricultural.finance.corporation",
        r"agriculture.production.report|national.agriculture.production",
        r"water.harvesting|irrigation.authority|nia.*strategic",
        r"catalogue.breeds",
        r"fsrp",
        r"dairy.interventions|daima",
        r"kasep.*policy",
        r"moalf|ministry.*agriculture.*strategic",
        r"dri.*annual|bri.*annual|hri.*annual|gerri.*annual|fcri.*annual|sri.*content",
        r"dvs.*magazine",
        r"december.*forecast|ond.*forecast|seasonal.*weather",
        r"national.*agriculture.*research",
        r"national.land.use.policy",
        r"eu.deforestation",
        r"regulatory.*impact.*assessment.*ria.*2022",
        r"migratory.*pests|invasive.*pests",
        r"consolidated.*summary.*ncce",
        r"nsafs.*facilitators",
        r"public.*consultation.*seeds",
        r"pyrethrum.*bill",
        r"ncpb.*national.*food",
        r"national.*cassava",
        r"avocado.*nairobi",
        r"grain.*feed.*annual",
        r"astgs",
        r"capacity.*building.*strategy.*agricultural",
        r"draft.*kasep",
        r"land.use.policy",
    ],
    "ict": [
        r"kictanet",
        r"kenya.*igf|igf.*kenya|iggf|eaigf|kigf",
        r"ca.*annual|communications.*authority.*annual",
        r"konza",
        r"kenya.*ai.*strategy",
        r"digital.*masterplan|ict.*masterplan|national.*ict.*masterplan",
        r"national.*ict.*policy|ict.*policy.*guidelines",
        r"national.*broadband",
        r"e.commerce.*strategy",
        r"national.*addressing",
        r"computer.*misuse|cybercrime|cybersecurity",
        r"data.*protection",
        r"mobile.*payments",
        r"ajira.*digital",
        r"finaccess|fin.*access",
        r"county.*ict.*survey",
        r"digital.*readiness",
        r"digital.*economy.*2019|kenya.*digital.*economy",
        r"optic.*fibre.*forensic|forensic.*optic.*fibre",
        r"invest.*kenya.*bpo",
        r"ncsc.*annual",
        r"africa.*tech.*policy",
        r"audience.*measurement",
        r"safer.*web.*women",
        r"smart.*cities",
        r"digital.*rights|upr.*digital",
        r"global.*geopolitics.*digital",
        r"digital.*skills.*africa",
        r"global.*digital.*health",
    ],
    "infrastructure": [
        r"kenha|kura|kerra|krc",
        r"rdm\.\d+\.\d+|road.*design.*manual",
        r"ketraco|kplc|kenya.*power|kengen",
        r"kpa|kenya.*ports",
        r"energy.*statistics|petroleum.*statistics|epra",
        r"sgr|standard.*gauge|railways",
        r"transport.*safety|ntsa",
        r"road.*safety|pavement|drainage.*design|bridge.*culvert",
        r"kosap|rural.*electrification",
        r"power.*sustainability",
        r"green.*mpa|escp.*green|sep.*green|lmp.*green",
        r"esmf.*green|statcom.*esia|esia.*addendum",
        r"bus.*rapid.*transport",
        r"intelligent.*transport",
        r"standard.*specification.*road",
        r"upgrading.*nanyuki|strengthening.*kandwia",
        r"regulatory.*impact.*assessment.*roads|kenya.*roads.*regulations",
        r"annual.*procurement.*plan.*kenha|maintenance.*capp",
        r"energy.*post|the.*energy.*post",
        r"iea.*kenya",
        r"kengen.*carbon",
        r"kenya.*gdp.*quarter|quarterly.*gdp",
        r"kenya.*leading.*economic.*indicators",
        r"financial.*statement.*\d{4}.*ketraco|ketraco.*financial",
        r"kerra.*service.*charter|service.*charter.*kerra",
        r"quality.*policy.*statement.*kenya.*railways|kr.*quality",
        r"alternative.*business.*partnerships",
        r"financialstatement|kpa.*financial",
    ],
    "anticorruption": [
        r"eacc",
        r"ppra|ppoa",
        r"odpp|director.*public.*prosecution",
        r"bribery.*act|bribery.*guidelines",
        r"anticorruption.*act|anti.*corruption.*act|anticorruptionrevised",
        r"public.*officer.*ethics",
        r"debarment",
        r"fatf.*recommendations",
        r"mer.*kenya|mutual.*evaluation",
        r"prosecution.*cec|unodc.*odpp",
        r"amendments.*regulations.*2020",
        r"ppada|public.*procurement.*asset.*disposal.*act",
        r"public.*procurement.*asset.*disposal",
        r"treasury.*memorandum.*public.*accounts",
    ],
    "president": [
        r"national.*values.*principles.*governance",
        r"election.*observer",
        r"constitution",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# ISSUING AGENT MAP
# ══════════════════════════════════════════════════════════════════════════════

ISSUING_AGENT_MAP = {
    "budget_policy_statement":   "finance",
    "budget_review_outlook":     "finance",
    "budget_summary":            "finance",
    "budget_speech":             "finance",
    "post_election_report":      "finance",
    "debt_management_strategy":  "finance",
    "public_debt_report":        "finance",
    "controller_of_budget":      "finance",
    "revenue_grants_estimates":  "finance",
    "statistical_annex":         "finance",
    "state_corporations_annex":  "finance",
    "pure_tables":               "finance",
    "cbk_annual_report":         "finance",
    "cbk_mpc_report":            "finance",
    "cbk_fsr_report":            "finance",
    "kra_revenue_performance":   "finance",
    "kra_corporate_plan":        "finance",
    "tax_expenditure_report":    "finance",
    "economic_survey":           "finance",
    "finance_act":               "finance",
    "finance_bill":              "finance",
    "imf_report":                "external",
    "world_bank_report":         "external",
    "auditor_general_report":    "anticorruption",
    "forensic_audit":            "anticorruption",
    "eacc_report":               "anticorruption",
    "ppra_report":               "anticorruption",
    "odpp_report":               "anticorruption",
    "mer_report":                "anticorruption",
    "audit_report":              "anticorruption",
    "igf_report":                "ict",
    "conference_report":         "president",
    "constitution":              "president",
    "guidelines":                "unknown",
    "manual":                    "unknown",
    "policy":                    "unknown",
    "strategic_plan":            "unknown",
    "annual_report":             "unknown",
    "research_report":           "unknown",
    "survey":                    "unknown",
    "bulletin":                  "unknown",
    "regulations":               "unknown",
    "act":                       "president",
    "bill":                      "president",
    "framework":                 "unknown",
    "assessment":                "unknown",
    "financial_statements":      "anticorruption",
    "statistics_report":         "unknown",
    "sector_report":             "unknown",
    "masterplan":                "unknown",
    "magazine":                  "unknown",
    "catalogue":                 "unknown",
    "unknown":                   "unknown",
}

ISSUING_AGENT_OVERRIDES = [
    (r"auditor.*general|auditor.gen|national.*government.*audit",   "anticorruption"),
    (r"eacc",                                                        "anticorruption"),
    (r"ppra|ppoa",                                                   "anticorruption"),
    (r"odpp",                                                        "anticorruption"),
    (r"forensic.*audit",                                             "anticorruption"),
    (r"konza.*financial|konza.*audited",                             "anticorruption"),
    (r"epra",                                                        "infrastructure"),
    (r"kictanet",                                                    "ict"),
    (r"ca.*annual|communications.*authority",                        "ict"),
    (r"kalro|kephis|afa",                                            "agriculture"),
    (r"kenha|kura|kerra|ketraco|kengen|kpa|kplc",                   "infrastructure"),
    (r"knec|kicd|tsc",                                               "education"),
    (r"cbk_|monetary.*policy.*committee",                            "finance"),
    (r"kra",                                                         "finance"),
    (r"imf",                                                         "external"),
    (r"world.*bank",                                                 "external"),
    (r"unodc|fatf",                                                  "external"),
    (r"wfp",                                                         "external"),
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT ACCESS TIERS
# ══════════════════════════════════════════════════════════════════════════════

UNIVERSAL_DOC_TYPES = {
    "budget_policy_statement",
    "economic_survey",
    "imf_report",
    "world_bank_report",
    "constitution",
    "finance_act",
    "finance_bill",
    "public_debt_report",
    "debt_management_strategy",
    "budget_review_outlook",
    "budget_summary",
    "statistical_annex",
    "revenue_grants_estimates",
    "post_election_report",
}

# FIX 2: None = primary_agents only (no president added)
#         Non-empty list = explicit shared access
#         Missing key = falls through to ADD_PRESIDENT_TO_ALL check below
SHARED_ACCESS_MAP = {
    "auditor_general_report":    ["anticorruption", "finance", "president"],
    "audit_report":              ["anticorruption", "finance", "president"],
    "forensic_audit":            ["anticorruption", "ict", "finance", "president"],
    "financial_statements":      ["anticorruption", "finance"],
    "kra_revenue_performance":   ["finance", "president"],
    "controller_of_budget":      ["finance", "president", "anticorruption"],
    "state_corporations_annex":  ["finance", "president"],
    "pure_tables":               ["finance"],
    "eacc_report":               ["anticorruption", "president", "finance"],
    "ppra_report":               ["anticorruption", "finance", "president"],
    "odpp_report":               ["anticorruption", "president"],
    "mer_report":                ["anticorruption", "finance", "president"],
    "igf_report":                ["ict", "president"],
    "cbk_annual_report":         ["finance"],
    "cbk_mpc_report":            ["finance"],
    "cbk_fsr_report":            ["finance"],
    "kra_corporate_plan":        ["finance"],
    "tax_expenditure_report":    ["finance"],
    "act":                       ["president", "anticorruption"],
    "bill":                      ["president", "anticorruption"],
    "regulations":               ["president", "anticorruption"],
    "conference_report":         ["president"],
    "survey":                    ["finance", "president"],
    "statistics_report":         ["finance", "president"],
    # None = primary_agents only, no auto-president
    "research_report":           None,
    "annual_report":             None,
    "bulletin":                  None,
    "guidelines":                None,
    "manual":                    None,
    "framework":                 None,
    "assessment":                None,
    "magazine":                  None,
    "catalogue":                 None,
}

# FIX 2: doc_types where president is always added even without explicit shared list
# strategic_plan, masterplan, policy, sector_report get president added automatically
ADD_PRESIDENT_TO_ALL = {
    "strategic_plan",
    "masterplan",
    "policy",
    "sector_report",
}

FILENAME_ACCESS_OVERRIDES = [
    (r"ngcdf",                       ["education", "anticorruption", "finance"]),
    (r"optic.*fibre.*forensic",      ["ict", "anticorruption", "president"]),
    (r"huduma.*namba",               ["ict", "president", "anticorruption"]),
    (r"election.*observer",          ["president", "anticorruption"]),
    (r"finaccess|fin.*access",       ["ict", "finance"]),
    (r"food.*loss|post.*harvest|food.*reserve", ["agriculture", "finance", "president"]),
    (r"county.*budget.*impl|county.*birr",      ["finance", "president", "anticorruption"]),
    (r"agricultural.*finance.*corp", ["agriculture", "finance"]),
    (r"land.*use.*policy",           ["agriculture", "president", "infrastructure"]),
    (r"youth.*agribusiness",         ["education", "agriculture"]),
    (r"digital.*skills.*africa",     ["ict", "education"]),
    (r"global.*digital.*health",     ["ict", "education", "finance"]),
    (r"kenya.*ai.*strategy",         ["ict", "president", "finance"]),
    (r"iea.*kenya|energy.*statistics|epra",     ["infrastructure", "finance", "president"]),
    (r"kenya.*gdp.*quarter|quarterly.*gdp",     ["infrastructure", "finance", "president"]),
    (r"kenya.*leading.*economic",               ["infrastructure", "finance", "president"]),
    (r"eacc.*vs|eacc.*julius",                  ["anticorruption", "president"]),
    (r"treasury.*memorandum",                   ["finance", "anticorruption", "president"]),
    (r"kenya.*roads.*regulations",              ["infrastructure", "president"]),
    (r"konza.*financial|konza.*audited",        ["ict", "anticorruption", "finance"]),
    (r"mer.*kenya|mutual.*evaluation",          ["anticorruption", "finance", "president"]),
    (r"unodc",                                  ["anticorruption", "president"]),
    (r"national.*values.*principles",           ["president", "anticorruption"]),
    (r"fsrp",                                   ["agriculture", "finance", "president"]),
    (r"eu.*deforestation",                      ["agriculture", "president", "ict"]),
]


# ══════════════════════════════════════════════════════════════════════════════
# TOPIC TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

AGENDA_FRAMES = {
    "vision_2030",
    "big_four",
    "beta_agenda",
    "devolution",
    "county_governance",
}

SUBJECT_TOPICS = {
    "public_debt", "fiscal_policy", "national_budget", "public_finance",
    "audit_compliance", "public_procurement", "governance",
    "monetary_policy", "tax_revenue", "tax_expenditure", "cbk_reports",
    "debt_management", "revenue_performance", "budget_implementation",
    "external_assessment", "financial_sector",
    "curriculum", "tvet", "ngcdf", "school_funding", "higher_education",
    "teacher_management", "education_policy",
    "food_security", "crop_research", "irrigation", "livestock",
    "agricultural_finance", "climate_agriculture", "seeds_regulations", "veterinary",
    "ict_policy", "cybersecurity", "digital_economy", "ai_policy",
    "internet_governance", "konza", "fintech", "digital_rights",
    "roads", "energy", "ports", "sgr", "infrastructure_finance",
    "road_design", "transport_policy", "rural_electrification",
    "eacc", "procurement_law", "forensic_audit", "anti_money_laundering", "prosecution",
    "youth_employment", "climate_resilience", "gender", "county_governance",
}

DOC_TYPE_TOPIC_MAP = {
    "budget_policy_statement":   ["fiscal_policy", "national_budget", "public_finance", "public_debt"],
    "budget_review_outlook":     ["fiscal_policy", "national_budget", "public_finance"],
    "budget_summary":            ["fiscal_policy", "national_budget"],
    "budget_speech":             ["fiscal_policy", "national_budget"],
    "post_election_report":      ["fiscal_policy", "public_finance"],
    "debt_management_strategy":  ["public_debt", "debt_management", "fiscal_policy"],
    "public_debt_report":        ["public_debt", "fiscal_policy"],
    "controller_of_budget":      ["budget_implementation", "fiscal_policy", "audit_compliance"],
    "revenue_grants_estimates":  ["fiscal_policy", "national_budget"],
    "statistical_annex":         ["fiscal_policy", "national_budget"],
    "state_corporations_annex":  ["fiscal_policy"],
    "pure_tables":               ["fiscal_policy"],
    "cbk_annual_report":         ["monetary_policy", "cbk_reports", "financial_sector"],
    "cbk_mpc_report":            ["monetary_policy", "cbk_reports"],
    "cbk_fsr_report":            ["monetary_policy", "cbk_reports", "financial_sector"],
    "kra_revenue_performance":   ["tax_revenue", "revenue_performance"],
    "kra_corporate_plan":        ["tax_revenue"],
    "tax_expenditure_report":    ["tax_expenditure", "tax_revenue"],
    "economic_survey":           ["fiscal_policy", "national_budget", "public_finance"],
    "finance_act":               ["fiscal_policy", "tax_revenue"],
    "finance_bill":              ["fiscal_policy", "tax_revenue"],
    "constitution":              ["governance", "public_finance", "devolution"],
    "imf_report":                ["external_assessment", "public_debt", "fiscal_policy"],
    "world_bank_report":         ["external_assessment", "fiscal_policy"],
    "auditor_general_report":    ["audit_compliance", "public_finance", "governance"],
    "audit_report":              ["audit_compliance", "governance"],
    "forensic_audit":            ["forensic_audit", "audit_compliance", "governance"],
    "financial_statements":      ["audit_compliance", "public_finance"],
    "eacc_report":               ["eacc", "governance", "audit_compliance"],
    "ppra_report":               ["procurement_law", "public_procurement", "governance"],
    "odpp_report":               ["prosecution", "governance"],
    "mer_report":                ["anti_money_laundering", "governance", "audit_compliance"],
    "igf_report":                ["internet_governance", "ict_policy", "digital_economy"],
    "annual_report":             [],
    "strategic_plan":            [],
    "policy":                    [],
    "sector_report":             [],
    "research_report":           [],
    "bulletin":                  [],
    "guidelines":                [],
    "manual":                    [],
    "framework":                 [],
    "assessment":                [],
    "survey":                    [],
    "regulations":               [],
    "act":                       ["governance"],
    "bill":                      ["governance"],
    "conference_report":         [],
    "statistics_report":         [],
    "masterplan":                [],
    "magazine":                  [],
    "catalogue":                 [],
    "unknown":                   [],
}

FILENAME_TOPIC_OVERRIDES = [
    (r"budget.*policy.*statement",          ["fiscal_policy", "national_budget"]),
    (r"economic.*survey",                   ["fiscal_policy", "national_budget"]),
    (r"public.*debt",                       ["public_debt", "debt_management"]),
    (r"cbk|monetary.*policy",              ["monetary_policy", "cbk_reports"]),
    (r"tax.*expenditure",                   ["tax_expenditure"]),
    (r"revenue.*performance|kra",           ["tax_revenue", "revenue_performance"]),
    (r"auditor.*general|audit.*national",   ["audit_compliance", "governance"]),
    (r"forensic.*audit",                    ["forensic_audit", "audit_compliance"]),
    (r"ngbirr|controller.*budget",         ["budget_implementation", "fiscal_policy"]),
    (r"imf",                                ["external_assessment", "public_debt"]),
    (r"world.*bank",                        ["external_assessment", "fiscal_policy"]),
    (r"constitution",                       ["governance", "public_finance"]),
    (r"finance.*act|finance.*bill",         ["fiscal_policy", "tax_revenue"]),
    (r"food.*security|food.*loss|post.*harvest", ["food_security"]),
    (r"kalro|crop.*conditions|cassava|avocado",  ["crop_research", "food_security"]),
    (r"irrigation|water.*harvesting|nia",        ["irrigation", "food_security"]),
    (r"livestock|veterinary|dvs",                ["livestock", "food_security"]),
    (r"agricultural.*finance.*corp",             ["agricultural_finance"]),
    (r"seeds|fertilizer|propagating",           ["seeds_regulations"]),
    (r"weather.*forecast|drought.*warning|dew",  ["climate_agriculture", "food_security"]),
    (r"agriculture.*insurance",                  ["agricultural_finance", "food_security"]),
    (r"eu.*deforestation",                       ["climate_agriculture", "food_security"]),
    (r"fsrp",                                    ["food_security", "agricultural_finance"]),
    (r"knec|kcse|kjsea",                         ["curriculum", "education_policy"]),
    (r"kicd|curriculum.*framework",              ["curriculum", "education_policy"]),
    (r"tvet",                                    ["tvet", "youth_employment"]),
    (r"ngcdf",                                   ["ngcdf", "school_funding"]),
    (r"capitation|school.*funding|school.*grant", ["school_funding"]),
    (r"teacher.*service|tsc",                    ["teacher_management"]),
    (r"higher.*education|helb|university",       ["higher_education"]),
    (r"youth.*agribusiness",                     ["youth_employment", "food_security"]),
    (r"digital.*skills.*africa",                 ["digital_economy", "youth_employment"]),
    (r"igf|internet.*governance",                ["internet_governance", "ict_policy"]),
    (r"kenya.*ai.*strategy",                     ["ai_policy", "digital_economy"]),
    (r"digital.*masterplan|ict.*masterplan",     ["ict_policy", "digital_economy"]),
    (r"cybersecurity|cybercrime|data.*protection", ["cybersecurity", "digital_rights"]),
    (r"konza",                                   ["konza", "digital_economy"]),
    (r"finaccess|mobile.*payments|fintech",      ["fintech", "digital_economy"]),
    (r"broadband|national.*ict",                 ["ict_policy", "digital_economy"]),
    (r"e.*commerce|ecommerce",                   ["digital_economy", "ict_policy"]),
    (r"ajira.*digital",                          ["digital_economy", "youth_employment"]),
    (r"digital.*rights|safer.*web",             ["digital_rights", "ict_policy"]),
    (r"huduma.*namba",                           ["digital_economy", "governance"]),
    (r"optic.*fibre",                            ["ict_policy", "forensic_audit"]),
    (r"kenha|kura|kerra|road.*design|rdm",       ["roads", "road_design"]),
    (r"ketraco|kplc|kengen|kenya.*power|energy", ["energy", "infrastructure_finance"]),
    (r"kpa|kenya.*ports",                        ["ports", "infrastructure_finance"]),
    (r"sgr|standard.*gauge|railways",            ["sgr", "infrastructure_finance"]),
    (r"epra|energy.*statistics|petroleum",       ["energy", "fiscal_policy"]),
    (r"ntsa|transport.*safety|road.*safety",     ["transport_policy", "roads"]),
    (r"kosap|rural.*electrification",            ["rural_electrification", "energy"]),
    (r"green.*mpa|bess.*project",                ["energy", "climate_resilience"]),
    (r"quarterly.*gdp|leading.*economic",        ["fiscal_policy", "infrastructure_finance"]),
    (r"eacc|ethics.*anti.*corruption",           ["eacc", "governance"]),
    (r"ppra|ppoa|procurement",                   ["procurement_law", "public_procurement"]),
    (r"odpp|director.*prosecution",              ["prosecution", "governance"]),
    (r"bribery|fatf|mer.*kenya",                 ["anti_money_laundering", "governance"]),
    (r"debarment",                               ["procurement_law", "public_procurement"]),
    (r"treasury.*memorandum",                    ["audit_compliance", "public_finance"]),
    (r"unodc",                                   ["prosecution", "governance"]),
    (r"devolution|county.*governance",           ["devolution", "county_governance"]),
    (r"vision.*2030|big.*four|beta.*agenda",     ["vision_2030", "big_four", "beta_agenda"]),
    (r"national.*values",                        ["governance", "county_governance"]),
    (r"land.*use.*policy",                       ["food_security", "infrastructure_finance"]),
    (r"climate.*resilience|climate.*adapt",      ["climate_resilience", "climate_agriculture"]),
    (r"gender|women",                            ["gender"]),
    (r"youth",                                   ["youth_employment"]),
]


# ══════════════════════════════════════════════════════════════════════════════
# COALITION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

COALITION_CONFIG = {
    "food_security": {
        "agents":        ["agriculture", "education", "finance", "president"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "youth_employment": {
        "agents":        ["education", "ict", "agriculture", "finance"],
        "max_per_agent": 5,
        "priority_floor": None,
    },
    "climate_resilience": {
        "agents":        ["agriculture", "infrastructure", "finance", "president"],
        "max_per_agent": 5,
        "priority_floor": None,
    },
    "digital_economy": {
        "agents":        ["ict", "finance", "education", "president"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "ai_policy": {
        "agents":        ["ict", "finance", "president", "education"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "infrastructure_finance": {
        "agents":        ["infrastructure", "finance", "president"],
        "max_per_agent": 7,
        "priority_floor": None,
    },
    "public_debt": {
        "agents":        ["finance", "president", "infrastructure"],
        "max_per_agent": 8,
        "priority_floor": None,
    },
    "public_procurement": {
        "agents":        ["anticorruption", "finance", "president",
                          "infrastructure", "education", "agriculture", "ict"],
        "max_per_agent": 3,
        "priority_floor": "medium",
    },
    "audit_compliance": {
        "agents":        ["anticorruption", "finance", "president"],
        "max_per_agent": 7,
        "priority_floor": None,
    },
    "fiscal_policy": {
        "agents":        ["finance", "president", "anticorruption"],
        "max_per_agent": 8,
        "priority_floor": None,
    },
    # FIX 5: monetary_policy was missing — without this, CBK/inflation queries
    # fall back to scoped (Finance only), excluding IMF/World Bank docs
    # that are tagged external_assessment + monetary_policy.
    "monetary_policy": {
        "agents":        ["finance", "president"],
        "max_per_agent": 8,
        "priority_floor": None,
    },
    "county_governance": {
        "agents":        ["anticorruption", "finance", "president", "infrastructure"],
        "max_per_agent": 5,
        "priority_floor": None,
    },
    "agricultural_finance": {
        "agents":        ["agriculture", "finance", "president"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "school_funding": {
        "agents":        ["education", "finance", "anticorruption"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "ngcdf": {
        "agents":        ["education", "anticorruption", "finance"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "energy": {
        "agents":        ["infrastructure", "finance", "president"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "governance": {
        "agents":        ["anticorruption", "president", "finance"],
        "max_per_agent": 6,
        "priority_floor": None,
    },
    "external_assessment": {
        "agents":        ALL_AGENTS,
        "max_per_agent": 3,
        "priority_floor": "medium",
    },
    "anti_money_laundering": {
        "agents":        ["anticorruption", "finance", "president"],
        "max_per_agent": 7,
        "priority_floor": None,
    },
    # Agenda frames
    "vision_2030": {
        "agents":        ALL_AGENTS,
        "max_per_agent": 3,
        "priority_floor": "high",
        "mode":          "open",
    },
    "big_four": {
        "agents":        ALL_AGENTS,
        "max_per_agent": 3,
        "priority_floor": "high",
        "mode":          "open",
    },
    "beta_agenda": {
        "agents":        ALL_AGENTS,
        "max_per_agent": 3,
        "priority_floor": "high",
        "mode":          "open",
    },
    "devolution": {
        "agents":        ALL_AGENTS,
        "max_per_agent": 3,
        "priority_floor": "high",
        "mode":          "open",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_MAP = {
    "budget_policy_statement":   "fiscal_policy",
    "budget_review_outlook":     "fiscal_policy",
    "budget_summary":            "fiscal_policy",
    "budget_speech":             "fiscal_policy",
    "post_election_report":      "fiscal_policy",
    "debt_management_strategy":  "fiscal_policy",
    "public_debt_report":        "fiscal_policy",
    "controller_of_budget":      "fiscal_policy",
    "revenue_grants_estimates":  "fiscal_policy",
    "statistical_annex":         "fiscal_policy",
    "state_corporations_annex":  "fiscal_policy",
    "pure_tables":               "fiscal_policy",
    "cbk_annual_report":         "monetary_policy",
    "cbk_mpc_report":            "monetary_policy",
    "cbk_fsr_report":            "monetary_policy",
    "kra_revenue_performance":   "revenue_tax",
    "kra_corporate_plan":        "revenue_tax",
    "tax_expenditure_report":    "tax_expenditure",
    "economic_survey":           "macroeconomic_data",
    "finance_act":               "legal_fiscal",
    "finance_bill":              "legal_fiscal",
    "constitution":              "constitutional",
    "imf_report":                "external_assessment",
    "world_bank_report":         "external_assessment",
    "auditor_general_report":    "audit_compliance",
    "audit_report":              "audit_compliance",
    "forensic_audit":            "audit_compliance",
    "financial_statements":      "audit_compliance",
    "eacc_report":               "governance",
    "ppra_report":               "procurement",
    "odpp_report":               "legal_compliance",
    "mer_report":                "governance",
    "sector_report":             "sector_policy",
    "statistics_report":         "sector_data",
    "framework":                 "sector_policy",
    "igf_report":                "internet_governance",
    "masterplan":                "sector_policy",
    "strategic_plan":            "sector_policy",
    "annual_report":             "institutional",
    "policy":                    "sector_policy",
    "research_report":           "sector_research",
    "survey":                    "sector_data",
    "bulletin":                  "sector_data",
    "guidelines":                "sector_policy",
    "manual":                    "sector_policy",
    "assessment":                "sector_research",
    "regulations":               "legal_fiscal",
    "act":                       "legal_fiscal",
    "bill":                      "legal_fiscal",
    "conference_report":         "sector_research",
    "magazine":                  "institutional",
    "catalogue":                 "institutional",
    "unknown":                   "unknown",
}


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING STRATEGY MAPS
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "budget_policy_statement":   1,
    "budget_review_outlook":     1,
    "budget_summary":            1,
    "budget_speech":             1,
    "post_election_report":      1,
    "debt_management_strategy":  1,
    "public_debt_report":        1,
    "statistical_annex":         2,
    "revenue_grants_estimates":  2,
    "state_corporations_annex":  2,
    "pure_tables":               2,
    "controller_of_budget":      3,
    "cbk_annual_report":         4,
    "cbk_mpc_report":            4,
    "cbk_fsr_report":            4,
    "finance_act":               5,
    "finance_bill":              5,
    "constitution":              5,
    "act":                       5,
    "bill":                      5,
    "regulations":               5,
    "auditor_general_report":    6,
    "audit_report":              6,
    "forensic_audit":            6,
    "eacc_report":               6,
    "ppra_report":               6,
    "financial_statements":      6,
    "kra_revenue_performance":   7,
    "kra_corporate_plan":        7,
    "tax_expenditure_report":    7,
    "economic_survey":           8,
    "imf_report":                8,
    "world_bank_report":         8,
}

CHUNKING_STRATEGY_MAP = {
    0: "narrative",
    1: "narrative",
    2: "tables_only",
    3: "narrative",
    4: "narrative",
    5: "legal",
    6: "audit_findings",
    7: "narrative",
    8: "hybrid",
}

CHUNK_SIZE_MAP    = {0:350, 1:350, 2:200, 3:400, 4:350, 5:250, 6:300, 7:350, 8:350}
CHUNK_OVERLAP_MAP = {0:50,  1:50,  2:0,   3:75,  4:50,  5:100, 6:75,  7:50,  8:50 }
CHUNK_MIN_TOKENS  = 100
CHUNK_MAX_TOKENS  = 500


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY AND RAG WEIGHT MAPS
# ══════════════════════════════════════════════════════════════════════════════

PRIORITY_MAP = {
    "constitution":              "constitutional",
    "finance_act":               "constitutional",
    "finance_bill":              "constitutional",
    "act":                       "high",
    "bill":                      "high",
    "regulations":               "medium",
    "budget_policy_statement":   "high",
    "budget_summary":            "high",
    "debt_management_strategy":  "high",
    "cbk_mpc_report":            "high",
    "statistical_annex":         "high",
    "revenue_grants_estimates":  "high",
    "pure_tables":               "high",
    "economic_survey":           "high",
    "budget_review_outlook":     "medium",
    "public_debt_report":        "medium",
    "auditor_general_report":    "medium",
    "audit_report":              "medium",
    "forensic_audit":            "medium",
    "financial_statements":      "medium",
    "kra_revenue_performance":   "medium",
    "tax_expenditure_report":    "medium",
    "cbk_annual_report":         "medium",
    "cbk_fsr_report":            "medium",
    "controller_of_budget":      "medium",
    "imf_report":                "medium",
    "world_bank_report":         "medium",
    "eacc_report":               "medium",
    "ppra_report":               "medium",
    "odpp_report":               "medium",
    "mer_report":                "medium",
    "igf_report":                "medium",
    "sector_report":             "medium",
    "strategic_plan":            "medium",
    "masterplan":                "medium",
    "policy":                    "medium",
    "survey":                    "medium",
    "statistics_report":         "medium",
    "annual_report":             "low",
    "kra_corporate_plan":        "low",
    "budget_speech":             "low",
    "post_election_report":      "medium",
    "state_corporations_annex":  "low",
    "research_report":           "low",
    "bulletin":                  "low",
    "guidelines":                "low",
    "manual":                    "low",
    "framework":                 "low",
    "assessment":                "low",
    "conference_report":         "low",
    "magazine":                  "low",
    "catalogue":                 "low",
    "unknown":                   "low",
}

RAG_WEIGHT_MAP = {
    "constitutional": 2.0,
    "high":           1.5,
    "medium":         1.0,
    "low":            0.5,
}

HAS_TABLES_MAP = {
    "budget_policy_statement":   True,
    "budget_review_outlook":     True,
    "budget_summary":            True,
    "debt_management_strategy":  True,
    "public_debt_report":        True,
    "controller_of_budget":      True,
    "economic_survey":           True,
    "kra_revenue_performance":   True,
    "tax_expenditure_report":    True,
    "auditor_general_report":    True,
    "audit_report":              True,
    "forensic_audit":            True,
    "financial_statements":      True,
    "cbk_annual_report":         True,
    "cbk_fsr_report":            True,
    "cbk_mpc_report":            False,
    "statistical_annex":         True,
    "revenue_grants_estimates":  True,
    "state_corporations_annex":  True,
    "pure_tables":               True,
    "finance_act":               False,
    "finance_bill":              False,
    "constitution":              False,
    "regulations":               False,
    "act":                       False,
    "bill":                      False,
    "imf_report":                True,
    "world_bank_report":         True,
    "statistics_report":         True,
    "survey":                    True,
    "eacc_report":               True,
    "ppra_report":               True,
    "mer_report":                True,
    "sector_report":             True,
    "strategic_plan":            False,
    "annual_report":             True,
    "research_report":           False,
    "igf_report":                False,
    "conference_report":         False,
    "bulletin":                  False,
    "guidelines":                False,
    "manual":                    False,
    "framework":                 False,
    "assessment":                False,
    "masterplan":                False,
    "magazine":                  False,
    "catalogue":                 False,
    "unknown":                   False,
}

# FIX 4: All doc_types now have explicit entries — no silent None returns.
# Callers still do .get(doc_type, []) as a safety guard, but every type is covered.
SKIP_SECTIONS_MAP = {
    "budget_policy_statement":  ["foreword", "acknowledgement", "table of contents"],
    "budget_review_outlook":    ["foreword", "acknowledgement", "table of contents"],
    "budget_summary":           [],
    "budget_speech":            [],
    "post_election_report":     ["foreword", "acknowledgement"],
    "debt_management_strategy": ["foreword", "acknowledgement"],
    "public_debt_report":       ["foreword", "acknowledgement"],
    "controller_of_budget":     ["county breakdown", "appendix", "foreword", "acknowledgement"],
    "revenue_grants_estimates": [],
    "statistical_annex":        [],
    "state_corporations_annex": [],
    "pure_tables":              [],
    "auditor_general_report":   ["foreword", "table of contents"],
    "audit_report":             ["foreword", "table of contents"],
    "forensic_audit":           ["foreword", "table of contents"],
    "financial_statements":     ["foreword", "table of contents", "notes to financial statements"],
    "cbk_annual_report": [
        "directors report", "directors' report",
        "financial statements", "statement of financial position",
        "income statement", "cash flow statement",
        "notes to financial statements", "staff costs",
        "human resources", "corporate governance", "board committees",
    ],
    "cbk_mpc_report":           ["foreword", "table of contents"],
    "cbk_fsr_report":           ["foreword", "table of contents"],
    "imf_report":               ["foreword", "acknowledgement"],
    "world_bank_report":        ["foreword", "acknowledgement"],
    "economic_survey":          ["foreword", "acknowledgement"],
    "kra_revenue_performance":  ["foreword", "acknowledgement"],
    "kra_corporate_plan":       ["foreword", "acknowledgement"],
    "tax_expenditure_report":   ["foreword", "acknowledgement"],
    "eacc_report":              ["foreword", "acknowledgement", "table of contents"],
    "ppra_report":              ["foreword", "acknowledgement", "table of contents"],
    "odpp_report":              ["foreword", "acknowledgement", "table of contents"],
    "mer_report":               ["foreword", "acknowledgement", "table of contents"],
    "strategic_plan":           ["foreword", "acknowledgement", "table of contents"],
    "masterplan":               ["foreword", "acknowledgement", "table of contents"],
    "annual_report":            ["foreword", "acknowledgement", "table of contents"],
    "policy":                   ["foreword", "acknowledgement"],
    "sector_report":            ["foreword", "acknowledgement", "table of contents"],
    "igf_report":               ["foreword", "acknowledgement", "table of contents"],
    "statistics_report":        ["foreword", "table of contents"],
    "research_report":          ["foreword", "acknowledgement"],
    "survey":                   ["foreword", "acknowledgement"],
    "bulletin":                 [],
    "guidelines":               ["foreword", "acknowledgement"],
    "manual":                   ["foreword", "acknowledgement", "table of contents"],
    "framework":                ["foreword", "acknowledgement", "table of contents"],
    "assessment":               ["foreword", "acknowledgement"],
    "conference_report":        ["foreword", "acknowledgement", "table of contents"],
    "regulations":              [],
    "act":                      [],
    "bill":                     [],
    "finance_act":              [],
    "finance_bill":             [],
    "constitution":             [],
    # FIX 4: magazine-specific skip list — removes non-content sections
    "magazine":                 ["table of contents", "advertisement", "editor's note",
                                 "from the desk", "letters to the editor"],
    "catalogue":                [],
    "unknown":                  [],
}


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL OVERRIDES
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_OVERRIDES = {
    "First Half NGBIRR FY 23-24 - COB final 13.3.14.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2023_24",
        "report_period": "h1",
    },
    "NGBIRR for First-Six Months of FY 2024-28.2..25 draft.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2024_25",
        "report_period": "h1",
    },
    "The-9th-Corporate-Plan.pdf": {
        "document_type": "kra_corporate_plan",
        "fiscal_year":   "2023_24",
        "report_period": "annual",
    },
    "NATIONAL-BOOK-web-1.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2019_20",
        "report_period": "annual",
    },
    "NATIONAL-GOVERNMENT-OCT-WEBSITE.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2021_22",
        "report_period": "annual",
    },
    "NATIONAL-GOVERNMENT-OCT-WEBSITE-1.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2022_23",
        "report_period": "annual",
    },
    "CBK_34th.pdf": {
        "document_type": "cbk_mpc_report",
        "fiscal_year":   "2023_24",
        "report_period": "h1",
    },
    "Kenya-Economic-Update-18-FINAL World bank.pdf": {
        "document_type": "world_bank_report",
        "fiscal_year":   "2017_18",
        "report_period": "annual",
    },
    "PUBLIC-KenyaEconomicUpdateFINAL World bank.pdf": {
        "document_type": "world_bank_report",
        "fiscal_year":   "2016_17",
        "report_period": "annual",
    },
    "Office of the Controller of Budget;.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2018_19",
        "report_period": "annual",
    },
    "IMF.pdf": {
        "document_type": "imf_report",
        "fiscal_year":   "2019_20",
        "report_period": "annual",
    },
    "CBK_2017 Annual Report.pdf": {
        "document_type": "cbk_annual_report",
        "fiscal_year":   "2016_17",
        "report_period": "annual",
    },
    "CBK_2018 Annual Report.pdf": {
        "document_type": "cbk_annual_report",
        "fiscal_year":   "2017_18",
        "report_period": "annual",
    },
    "Medium-Term-Debt-Management-Strategy-2022.pdf": {
        "document_type": "debt_management_strategy",
        "fiscal_year":   "2022_23",
        "report_period": "annual",
    },
    "NGBIRR Book Report May 2025 a.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2024_25",
        "report_period": "annual",
    },
    "7th-Corporate-Plan-FA-Online-version-min.pdf": {
        "document_type": "kra_corporate_plan",
        "fiscal_year":   "2017_18",
        "report_period": "annual",
    },
    "KRA-8TH-CORPORATE-PLAN-.pdf": {
        "document_type": "kra_corporate_plan",
        "fiscal_year":   "2020_21",
        "report_period": "annual",
    },
    "CBK_29th Monetary Policy Committee Report.pdf": {
        "document_type": "cbk_mpc_report",
        "fiscal_year":   "2021_22",
        "report_period": "h2",
    },
    "REVENUE-GRANTS-AND-LOANS-ESTIMATES-FINAL-pure tables.pdf": {
        "document_type": "revenue_grants_estimates",
        "fiscal_year":   "2020_21",
        "report_period": "annual",
    },
    "TheConstitutionOfKenya.pdf": {
        "document_type":  "constitution",
        "fiscal_year":    "na",
        "report_period":  "annual",
        "primary_agents": ["president", "finance", "anticorruption"],
        "agent_access":   ALL_AGENTS,
        "issuing_agent":  "president",
        "topics":         ["governance", "public_finance", "devolution", "county_governance"],
    },
    "Anuall Coprate Report-2022-23-1.pdf": {
        "document_type": "kra_corporate_plan",
        "fiscal_year":   "2022_23",
        "report_period": "annual",
    },
    "SUMMARY-REPORT-2019-2020.pdf": {
        "document_type": "auditor_general_report",
        "fiscal_year":   "2019_20",
        "report_period": "annual",
    },
    "CBK_28th Bi-Annual Report of the MPC April 2022.pdf": {
        "document_type": "cbk_mpc_report",
        "fiscal_year":   "2021_22",
        "report_period": "h1",
    },
    "CBK_43rd Monetary Policy Statement, December 2018.pdf": {
        "document_type": "cbk_mpc_report",
        "fiscal_year":   "2018_19",
        "report_period": "annual",
    },
    "2021-Budget-Policy-Statement.pdf": {
        "fiscal_year":   "2020_21",
        "report_period": "annual",
    },
    "2022-Budget-Policy-Statement.pdf": {
        "fiscal_year":   "2021_22",
        "report_period": "annual",
    },
    "2020 Budget Policy Statement.pdf": {
        "fiscal_year":   "2019_20",
        "report_period": "annual",
    },
    "2020-Magazine-DVS-A-New-Dawn.pdf": {
        "document_type":  "magazine",
        "primary_agents": ["agriculture"],
        "agent_access":   ["agriculture"],
        "issuing_agent":  "agriculture",
        "topics":         ["livestock", "veterinary"],
        "priority":       "low",
    },
    "2021-Final-Mag-Post.pdf": {
        "document_type":  "magazine",
        "primary_agents": ["infrastructure"],
        "agent_access":   ["infrastructure", "finance"],
        "issuing_agent":  "infrastructure",
        "topics":         ["energy"],
        "priority":       "low",
    },
    "2023-Energy-Post-SecondEdition.pdf": {
        "document_type":  "bulletin",
        "primary_agents": ["infrastructure"],
        "agent_access":   ["infrastructure", "finance", "president"],
        "issuing_agent":  "infrastructure",
        "topics":         ["energy"],
    },
    "2023-Election-Observer-Report.pdf": {
        "document_type":  "research_report",
        "primary_agents": ["president"],
        "agent_access":   ["president", "anticorruption"],
        "issuing_agent":  "anticorruption",
        "topics":         ["governance"],
        "priority":       "medium",
    },
    "2020-Annual-Report-Huduma-Namba.pdf": {
        "document_type":  "research_report",
        "primary_agents": ["president", "ict"],
        "agent_access":   ["president", "ict", "anticorruption"],
        "issuing_agent":  "president",
        "topics":         ["digital_economy", "governance"],
        "priority":       "medium",
    },

    # FIX 6: BPS superseded series
    # TODO: mark_superseded.py will walk BPS_TYPES sorted by doc_year and set
    # superseded=True on all but the latest per (doc_type, primary_agents) combo.
    # The 2026 BPS is currently the live version; all earlier BPS are historical.
    # DO NOT set superseded=True manually here — let mark_superseded.py handle it
    # so the logic is reproducible and not scattered across this file.
    "2026 Budget Policy Statement.pdf": {
        "superseded": False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FY DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

BPS_TYPES = {
    "budget_policy_statement",
    "budget_review_outlook",
    "budget_summary",
}

FORWARD_LOOKING_TYPES = {
    "debt_management_strategy",
}

COVER_FY_PATTERNS = [
    (r"financial year\s+(\d{4})[/\-](\d{2,4})",             "range"),
    (r"fy\s*(\d{4})[/\-](\d{2,4})",                         "range"),
    (r"f\.y\.?\s*(\d{4})[/\-](\d{2,4})",                    "range"),
    (r"1st july\s+(\d{4})\s+to\s+30th june\s+(\d{4})",      "range"),
    (r"july\s+(\d{4})\s+to\s+june\s+(\d{4})",               "range"),
    (r"ended\s+30\s+june\s+(\d{4})",                         "end_year"),
    (r"ending\s+june\s+(\d{4})",                             "end_year"),
    (r"ended\s+june\s+(\d{4})",                              "end_year"),
    (r"for\s+the\s+period.*?(\d{4}).*?(\d{4})",              "range"),
    (r"(\d{4})[/\-](\d{2,4})\s+(?:budget|fiscal|annual)",   "range"),
    (r"(?:budget|fiscal|annual)\s+(\d{4})[/\-](\d{2,4})",   "range"),
]

COVER_PERIOD_PATTERNS = [
    (r"first\s+quarter|q1\b|first\s+three\s+months",         "q1"),
    (r"first\s+half|first\s+six\s+months|h1\b",              "h1"),
    (r"second\s+half|second\s+six\s+months|h2\b",            "h2"),
    (r"nine\s+months|third\s+quarter",                        "q3"),
    (r"mid.?year\s+review",                                   "mid_year"),
    (r"\bannual\s+report\b|\bfull\s+year\b",                  "annual"),
]


# ══════════════════════════════════════════════════════════════════════════════
# VALID DOMAINS
# ══════════════════════════════════════════════════════════════════════════════

VALID_DOMAINS = {
    "fiscal_policy", "monetary_policy", "audit_compliance", "revenue_tax",
    "tax_expenditure", "macroeconomic_data", "legal_fiscal", "constitutional",
    "external_assessment", "governance", "procurement", "legal_compliance",
    "internet_governance", "sector_policy", "sector_data", "sector_research",
    "institutional", "unknown",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_slug(name: str) -> str:
    """Convert filename to normalised slug for pattern matching."""
    from pathlib import Path
    stem = Path(name).stem if name.lower().endswith(".pdf") else name
    return re.sub(r"[\s\-\.]", ".", stem).lower()


def match_doc_type(slug: str) -> str:
    """Return first matching doc_type from DOCUMENT_TYPE_RULES or 'unknown'."""
    for pattern, doc_type in DOCUMENT_TYPE_RULES:
        if re.search(pattern, slug):
            return doc_type
    return "unknown"


def match_primary_agents(slug: str, doc_type: str) -> list:
    """
    Return list of primary agents from AGENT_PATTERNS.

    FIX 1: Warns when >2 agents match — likely a pattern collision rather than a
    genuine joint-ministry document. Logs to unknown_doc_types.log for review.
    Callers should inspect the log before upserting any agent's corpus.
    """
    matched = []
    for agent, patterns in AGENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, slug):
                if agent not in matched:
                    matched.append(agent)
                break

    if not matched:
        matched = ["president"]  # safe fallback

    if len(matched) > 2:
        msg = f"COLLISION: {slug!r} matched {len(matched)} agents: {matched}"
        logger.warning(msg)
        try:
            with open("unknown_doc_types.log", "a") as f:
                f.write(f"COLLISION\t{slug}\t{matched}\n")
        except OSError:
            pass  # don't crash the pipeline over a log write

    return matched


def match_issuing_agent(slug: str, doc_type: str) -> str:
    """Return issuing agent from overrides then doc_type map."""
    for pattern, agent in ISSUING_AGENT_OVERRIDES:
        if re.search(pattern, slug):
            return agent
    return ISSUING_AGENT_MAP.get(doc_type, "unknown")


def build_agent_access(
    slug:           str,
    doc_type:       str,
    primary_agents: list,
    issuing_agent:  str,
) -> list:
    """
    Build full agent_access list from tiers + filename overrides.

    FIX 2: Replaces the silent empty-list "primary + president" convention with:
      - None in SHARED_ACCESS_MAP  → primary_agents only, no auto-president
      - Non-empty list             → explicit shared access
      - doc_type in ADD_PRESIDENT_TO_ALL → president always added
    This makes access grants visible in the data, not in a comment.
    """
    access = set(primary_agents)

    # Tier 1: universal — everyone gets these doc_types
    if doc_type in UNIVERSAL_DOC_TYPES:
        access.update(ALL_AGENTS)
        return sorted(access)

    # Tier 2: shared access map
    shared = SHARED_ACCESS_MAP.get(doc_type)   # None if key missing or None value
    if shared is not None:
        access.update(shared)

    # Explicit president-always set (replaces empty-list convention)
    if doc_type in ADD_PRESIDENT_TO_ALL:
        access.add("president")

    # Always add issuing agent (except external bodies and unknowns)
    if issuing_agent not in ("unknown", "external"):
        access.add(issuing_agent)

    # Tier 3: filename overrides
    for pattern, extra_agents in FILENAME_ACCESS_OVERRIDES:
        if re.search(pattern, slug):
            access.update(extra_agents)

    return sorted(access)


def build_topics(slug: str, doc_type: str) -> list:
    """Build topic list from doc_type base + filename overrides."""
    topics = set(DOC_TYPE_TOPIC_MAP.get(doc_type, []))
    for pattern, extra_topics in FILENAME_TOPIC_OVERRIDES:
        if re.search(pattern, slug):
            topics.update(extra_topics)
    return sorted(topics)


def extract_doc_year(filename: str) -> int | None:
    """
    Extract publication year as integer from filename.

    FIX 3: For YYYY-YYYY or YYYY-YY range filenames, returns the LATER year.
    Example: 'Annual-Public-Debt-Report-2022-2023.pdf' → 2023 (not 2022).
    This matters for recency ranking in the retriever.
    """
    # Check for year range first: YYYY-YYYY or YYYY-YY
    range_match = re.search(r'(20\d{2})[\-/](20)?(\d{2})\b', filename)
    if range_match:
        y1     = int(range_match.group(1))
        prefix = range_match.group(2)   # "20" or None
        suffix = range_match.group(3)   # 2-digit remainder
        if prefix:
            y2 = int(prefix + suffix)   # full 4-digit second year
        else:
            y2 = int(str(y1)[:2] + suffix)  # infer century from first year
        return max(y1, y2)

    # Single year
    single = re.search(r'(?<!\d)(20\d{2})(?!\d)', filename)
    return int(single.group(1)) if single else None