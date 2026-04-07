"""
test_taxonomy.py
Run from ~/PRES/Executive/
  conda activate ml-env
  python3 test_taxonomy.py
"""

import sys
sys.path.insert(0, "Common")

from doc_type_taxonomy import (
    normalize_slug, match_doc_type, match_primary_agents,
    match_issuing_agent, build_agent_access, build_topics, extract_doc_year,
    resolve_overrides, ALL_AGENTS,   # ← added resolve_overrides
)

# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    # Known edge cases from handoff doc
    ("2025-Strategy-GOK-Kenya-AI.pdf",                  "strategic_plan",       ["ict"]),
    ("CBK_34th.pdf",                                    "cbk_mpc_report",       ["finance"]),
    ("2025-Statistics-Report-KNEC-KCSE-Examination.pdf","statistics_report",    ["education"]),
    ("2024-Annual-Report-KICTANet.pdf",                 "annual_report",        ["ict"]),
    ("2022-Annual-Report-KALRO-Dairy-Research-Institute.pdf", "annual_report",  ["agriculture"]),

    # Finance core
    ("2024-Budget-Policy-Statement.pdf",                "budget_policy_statement", ["finance"]),
    ("TheConstitutionOfKenya.pdf",                      "constitution",         ["president", "finance", "anticorruption"]),
    ("2023-EACC-Annual-Report.pdf",                     "eacc_report",          ["anticorruption"]),
    ("2024-KeNHA-Annual-Report.pdf",                    "annual_report",        ["infrastructure"]),
    ("IMF.pdf",                                         "imf_report",           ["finance"]),
    ("2023-PPRA-Annual-Report.pdf",                     "ppra_report",          ["anticorruption"]),
    ("Annual-Public-Debt-Report-2022-2023.pdf",         "public_debt_report",   ["finance"]),

    # Year extraction edge cases
    ("2022-Budget-Review-Outlook-2023.pdf",             None,                   None),
    ("Medium-Term-Debt-Management-Strategy-2022.pdf",   "debt_management_strategy", ["finance"]),

    # Cross-cutting
    ("2024-Kenya-Digital-Economy-Strategy.pdf",         "strategic_plan",       ["ict"]),
    ("2023-KALRO-Seeds-Regulations.pdf",                "regulations",          ["agriculture"]),
    ("2024-KeNHA-Road-Design-Manual.pdf",               "manual",               ["infrastructure"]),
    ("2023-KNEC-KCSE-Statistics-Report.pdf",            "statistics_report",    ["education"]),
    ("2024-Finance-Act.pdf",                            "finance_act",          ["finance"]),
]

# ── Runner ────────────────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

passes = fails = warnings = 0

print(f"\n{'='*100}")
print(f"{'FILE':<55} {'DOC_TYPE':<28} {'PRIMARY':<30} {'ACCESS_COUNT':<14} {'TOPICS':<5} {'YEAR'}")
print(f"{'='*100}")

for filename, exp_dt, exp_pa in TEST_CASES:
    slug    = normalize_slug(filename)
    dt      = match_doc_type(slug)
    pa      = match_primary_agents(slug, dt)
    ia      = match_issuing_agent(slug, dt)
    access  = build_agent_access(slug, dt, pa, ia)
    topics  = build_topics(slug, dt)
    year    = extract_doc_year(filename)

    # === KEY FIX: Apply manual overrides (exactly as tag_documents.py will do) ===
    meta = {
        "document_type": dt,
        "primary_agents": pa,
        "issuing_agent": ia,
        "agent_access": access,
        "topics": topics,
    }
    meta = resolve_overrides(filename, meta)

    # Use the overridden values for assertions and printing
    dt = meta["document_type"]
    pa = meta.get("primary_agents", pa)
    access = meta.get("agent_access", access)
    topics = meta.get("topics", topics)

    # Assertions
    dt_ok = (exp_dt is None) or (dt == exp_dt)
    pa_ok = (exp_pa is None) or (sorted(pa) == sorted(exp_pa))
    status = PASS if (dt_ok and pa_ok) else FAIL

    if dt_ok and pa_ok:
        passes += 1
    else:
        fails += 1

    if dt == "unknown":
        warnings += 1
        status = WARN

    short_name = filename[:54]
    print(f"{short_name:<55} {dt:<28} {str(pa):<30} {len(access):<14} {len(topics):<5} {year}")

    if not dt_ok:
        print(f"  {FAIL} doc_type: got={dt!r}  expected={exp_dt!r}")
    if not pa_ok:
        print(f"  {FAIL} primary_agents: got={pa}  expected={exp_pa}")

print(f"{'='*100}")
print(f"\nResults: {passes} passed  |  {fails} failed  |  {warnings} unknown doc_types\n")

# ── Access spot-checks ────────────────────────────────────────────────────────
print("── Access spot-checks ──────────────────────────────────────────────────")
spot_checks = [
    ("TheConstitutionOfKenya.pdf", ALL_AGENTS,                         "constitution → all agents"),
    ("2024-Budget-Policy-Statement.pdf", ALL_AGENTS,                   "BPS → all agents (universal)"),
    ("CBK_34th.pdf", ["finance"],                                      "CBK MPC → finance only"),
    ("2023-EACC-Annual-Report.pdf", ["anticorruption","finance","president"], "EACC → anticorruption+finance+president"),
    ("2025-Strategy-GOK-Kenya-AI.pdf", ["finance","ict","president"],  "Kenya AI strategy → ict+finance+president"),
]

for filename, expected_access, label in spot_checks:
    slug   = normalize_slug(filename)
    dt     = match_doc_type(slug)
    pa     = match_primary_agents(slug, dt)
    ia     = match_issuing_agent(slug, dt)
    access = build_agent_access(slug, dt, pa, ia)

    # Apply override here too for consistency
    meta = {"document_type": dt, "primary_agents": pa, "agent_access": access}
    meta = resolve_overrides(filename, meta)
    access = meta.get("agent_access", access)

    ok = all(a in access for a in expected_access)
    print(f"  {'OK' if ok else 'FAIL':<4} {label}")
    if not ok:
        missing = [a for a in expected_access if a not in access]
        print(f"       missing from access: {missing}")
        print(f"       got: {access}")

# ── Year extraction ───────────────────────────────────────────────────────────
print("\n── Year extraction ─────────────────────────────────────────────────────")
year_tests = [
    ("Annual-Public-Debt-Report-2022-2023.pdf", 2023),
    ("2022-Budget-Review-Outlook-2023.pdf",     2023),
    ("CBK_2017 Annual Report.pdf",              2017),
    ("Medium-Term-Debt-Management-Strategy-2022.pdf", 2022),
    ("2024-Budget-Policy-Statement.pdf",        2024),
    ("IMF.pdf",                                 None),
]
for filename, expected in year_tests:
    got = extract_doc_year(filename)
    ok  = got == expected
    print(f"  {'OK' if ok else 'FAIL':<4} {filename:<50} got={got}  expected={expected}")

print()