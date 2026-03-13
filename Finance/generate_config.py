"""
generate_config.py — Auto-generate config.yaml for Kenya AI Executive Roundtable
RAG pipeline only — no training data generation.

Detection runs in 3 passes:
  Pass 1 — Filename patterns
  Pass 2 — Cover text (first 3000 chars) — only if Pass 1 left gaps
  Pass 3 — Deep text (chars 3000-10000) — only if FY still unknown after Pass 2
  Final  — MANUAL_OVERRIDES always win

Usage:
    python generate_config.py
    python generate_config.py --raw-dir /path/to/data/raw --no-pdf

Output:
    config.yaml in the same directory as this script
"""

import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
except ImportError:
    raise ImportError("Run: pip install pyyaml")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️  PyMuPDF not found — PDF cover text detection disabled.")
    print("   Run: pip install pymupdf  to enable it.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL OVERRIDES — applied last, always win over all detection passes
# Add entries here for genuinely undatable files after running the script
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
    "Annual-Public-Debt-Management-Report-.pdf": {
        "document_type": "public_debt_report",
        "fiscal_year":   "2017_18",
        "report_period": "annual",
    },
    "IMF.pdf": {
        "document_type": "imf_report",
        "fiscal_year":   "2019_20",
        "report_period": "annual",
    },
    # Cover text pulling wrong year (publication year confusion)
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
    # Cover text grabbed a forward projection year instead of report year
    "Medium-Term-Debt-Management-Strategy-2022.pdf": {
        "document_type": "debt_management_strategy",
        "fiscal_year":   "2022_23",
        "report_period": "annual",
    },
    # Annual report misdetected as q3 from cover text
    "NGBIRR Book Report May 2025 a.pdf": {
        "document_type": "controller_of_budget",
        "fiscal_year":   "2024_25",
        "report_period": "annual",
    },

    # Hard flags — FY unknown after all passes
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
    # "na" = genuinely has no fiscal year, not a detection failure
    # Downstream pipeline must treat "na" as timeless — no recency filtering
    "TheConstitutionOfKenya.pdf": {
        "document_type": "constitution",
        "fiscal_year":   "na",
        "report_period": "annual",
    },

    # Hard flags — document type unknown after all passes
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

    # Misclassified by cover text — MPC report not annual report
    "CBK_28th Bi-Annual Report of the MPC April 2022.pdf": {
        "document_type": "cbk_mpc_report",
        "fiscal_year":   "2021_22",
        "report_period": "h1",
    },
    # Out-of-cycle December statement — numbered 43rd breaks even/odd scheme
    "CBK_43rd Monetary Policy Statement, December 2018.pdf": {
        "document_type": "cbk_mpc_report",
        "fiscal_year":   "2018_19",
        "report_period": "annual",
    },
    # Cover text picked up a period reference from body — BPS is always annual
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
}

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION RULES
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENT_TYPE_RULES = [
    (r"pure.?table",                                        "pure_tables"),
    (r"statistical.?annex",                                 "statistical_annex"),
    (r"estimates.?revenue.?grants",                         "revenue_grants_estimates"),
    (r"revenue.?grants.?loans",                             "revenue_grants_estimates"),
    (r"revenue.?grants.?and.?loans",                        "revenue_grants_estimates"),
    (r"annex.?state.?corp",                                 "state_corporations_annex"),
    (r"budget.?policy.?statement|bps",                      "budget_policy_statement"),
    (r"budget.*review.*outlook|brop",                       "budget_review_outlook"),
    (r"budget.?summary",                                    "budget_summary"),
    (r"budget.?speech",                                     "budget_speech"),
    (r"post.?election.?economic",                           "post_election_report"),
    (r"budget.?statement",                                  "budget_statement"),
    (r"medium.?term.?debt|mtds",                            "debt_management_strategy"),
    (r"annual.?public.?debt|public.?debt.?report",          "public_debt_report"),
    (r"annual.?public.?debt.?management",                   "public_debt_report"),
    (r"ngbirr",                                             "controller_of_budget"),
    (r"annual.?birr|ng.?annual.?birr",                      "controller_of_budget"),
    (r"national.?government.?budget.?impl",                 "controller_of_budget"),
    (r"controller.?of.?budget",                             "controller_of_budget"),
    (r"national.?government.?oct",                          "controller_of_budget"),
    (r"national.?book",                                     "controller_of_budget"),
    (r"national.?budget.?may",                              "controller_of_budget"),
    (r"national.?report.?book",                             "controller_of_budget"),
    (r"september.?book",                                    "controller_of_budget"),
    (r"national.?government.?book",                         "controller_of_budget"),
    (r"annual.?national.?government.?budget",               "controller_of_budget"),
    (r"auditor.?general|auditor.?gen",                      "auditor_general_report"),
    (r"summary.?report.?auditor",                           "auditor_general_report"),
    (r"national.?government.?audit",                        "auditor_general_report"),
    (r"report.?auditor.?general",                           "auditor_general_report"),
    (r"summary.?report.*national.*government",              "auditor_general_report"),
    (r"cbk.*annual.*report|cbk_\d{4}|cnk.*annual",         "cbk_annual_report"),
    (r"cbk.*mpc|monetary.?policy.?comm",                    "cbk_mpc_report"),
    (r"cbk.*fsr|financial.?sector.?stab",                   "cbk_fsr_report"),
    (r"monetary.?policy.?statement",                        "cbk_mpc_report"),
    (r"annual.?revenue.?performance",                       "kra_revenue_performance"),
    (r"revenue.?performance",                               "kra_revenue_performance"),
    (r"annual.?corporate.?report|corporate.?report",        "kra_corporate_plan"),
    (r"kra.*corporate|corporate.*plan.*kra",                "kra_corporate_plan"),
    (r"tax.?expenditure",                                   "tax_expenditure_report"),
    (r"kra.*plan|corporate.*plan",                          "kra_corporate_plan"),
    (r"economic.?survey",                                   "economic_survey"),
    (r"finance.*act",                                       "finance_act"),
    (r"finance.*bill",                                      "finance_bill"),
    (r"constitution",                                       "constitution"),
    (r"imf",                                                "imf_report"),
    (r"world.?bank|kenya.?economic.?update",                "world_bank_report"),
]

# Cover text type patterns — used only when filename gives unknown
# Less aggressive than filename rules to avoid false positives
COVER_TYPE_RULES = [
    (r"budget.*policy.*statement",                          "budget_policy_statement"),
    (r"budget.*review.*outlook|budget.*outlook.*paper",     "budget_review_outlook"),
    (r"controller of budget",                               "controller_of_budget"),
    (r"budget implementation review",                       "controller_of_budget"),
    (r"annual corporate report",                            "kra_corporate_plan"),
    (r"kenya revenue authority.*annual",                    "kra_corporate_plan"),
    (r"auditor.general.*national government",               "auditor_general_report"),
    (r"report of the auditor",                              "auditor_general_report"),
    (r"monetary policy committee",                          "cbk_mpc_report"),
    (r"financial sector.*stability",                        "cbk_fsr_report"),
    (r"central bank of kenya.*annual report",               "cbk_annual_report"),
    (r"annual report.*central bank of kenya",               "cbk_annual_report"),
    (r"medium.term debt management",                        "debt_management_strategy"),
    (r"public debt.*management.*report",                    "public_debt_report"),
    (r"annual public debt",                                 "public_debt_report"),
    (r"economic survey",                                    "economic_survey"),
    (r"international monetary fund|imf.*kenya|kenya.*imf",  "imf_report"),
    (r"world bank.*kenya|kenya.*economic update",           "world_bank_report"),
    (r"revenue.*performance.*report",                       "kra_revenue_performance"),
    (r"tax expenditure report",                             "tax_expenditure_report"),
]

# FY patterns for cover text — ordered by specificity
COVER_FY_PATTERNS = [
    # Explicit FY notation
    (r"financial year\s+(\d{4})[/\-](\d{2,4})",            "range"),
    (r"fy\s*(\d{4})[/\-](\d{2,4})",                        "range"),
    (r"f\.y\.?\s*(\d{4})[/\-](\d{2,4})",                   "range"),
    # Period stated as dates
    (r"1st july\s+(\d{4})\s+to\s+30th june\s+(\d{4})",     "range"),
    (r"july\s+(\d{4})\s+to\s+june\s+(\d{4})",              "range"),
    (r"ended\s+30\s+june\s+(\d{4})",                        "end_year"),
    (r"ending\s+june\s+(\d{4})",                            "end_year"),
    (r"ended\s+june\s+(\d{4})",                             "end_year"),
    (r"for\s+the\s+period.*?(\d{4}).*?(\d{4})",             "range"),
    # Year range in title
    (r"(\d{4})[/\-](\d{2,4})\s+(?:budget|fiscal|annual|report|review)", "range"),
    (r"(?:budget|fiscal|annual|report|review)\s+(\d{4})[/\-](\d{2,4})", "range"),
]

# Period patterns for cover text
COVER_PERIOD_PATTERNS = [
    (r"first\s+quarter|q1\b|first\s+three\s+months",        "q1"),
    (r"first\s+half|first\s+six\s+months|h1\b",             "h1"),
    (r"second\s+half|second\s+six\s+months|h2\b",           "h2"),
    (r"nine\s+months|third\s+quarter",                       "q3"),
    (r"mid.?year\s+review",                                  "mid_year"),
    (r"\bannual\s+report\b|\bfull\s+year\b",                 "annual"),
]

# Supersedes hints — output treated as soft only, never hard-resolved
SUPERSEDES_HINT_PATTERNS = [
    r"revised\s+(?:estimates|figures|projections)",
    r"updates?\s+(?:the\s+)?(?:figures|projections|estimates)",
    r"replaces?\s+the\s+(?:earlier|previous|original)",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "budget_policy_statement":   1,
    "budget_review_outlook":     1,
    "budget_summary":            1,
    "budget_speech":             1,
    "budget_statement":          1,
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
    "auditor_general_report":    6,
    "kra_revenue_performance":   7,
    "kra_corporate_plan":        7,
    "tax_expenditure_report":    7,
    "economic_survey":           8,
    "imf_report":                8,
    "world_bank_report":         8,
    "unknown":                   0,
}

CHUNKING_STRATEGY_MAP = {
    1: "narrative",   2: "tables_only",  3: "narrative",
    4: "narrative",   5: "legal",        6: "audit_findings",
    7: "narrative",   8: "hybrid",       0: "narrative",
}

CHUNK_BY_MAP = {
    1: "paragraph",  2: "row",       3: "paragraph",
    4: "paragraph",  5: "clause",    6: "finding",
    7: "paragraph",  8: "paragraph", 0: "token",
}

CHUNK_SIZE_MAP = {
    1: 350,  2: 200,  3: 400,
    4: 350,  5: 250,  6: 300,
    7: 350,  8: 350,  0: 350,
}

CHUNK_OVERLAP_MAP = {
    1: 50,   2: 0,    3: 75,
    4: 50,   5: 100,  6: 75,
    7: 50,   8: 50,   0: 50,
}

CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 500

DOMAIN_MAP = {
    "budget_policy_statement":   "fiscal_policy",
    "budget_review_outlook":     "fiscal_policy",
    "budget_summary":            "fiscal_policy",
    "budget_speech":             "fiscal_policy",
    "budget_statement":          "fiscal_policy",
    "post_election_report":      "fiscal_policy",
    "debt_management_strategy":  "fiscal_policy",
    "public_debt_report":        "fiscal_policy",
    "controller_of_budget":      "fiscal_policy",
    "auditor_general_report":    "audit_compliance",
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
    "statistical_annex":         "fiscal_policy",
    "revenue_grants_estimates":  "fiscal_policy",
    "state_corporations_annex":  "fiscal_policy",
    "pure_tables":               "fiscal_policy",
    "imf_report":                "external_assessment",
    "world_bank_report":         "external_assessment",
    "unknown":                   "unknown",
}

PRIORITY_MAP = {
    "constitution":              "constitutional",
    "finance_act":               "constitutional",
    "finance_bill":              "constitutional",
    "budget_policy_statement":   "high",
    "budget_summary":            "high",
    "debt_management_strategy":  "high",
    "cbk_mpc_report":            "high",
    "budget_review_outlook":     "medium",
    "public_debt_report":        "medium",
    "auditor_general_report":    "medium",
    "economic_survey":           "medium",
    "kra_revenue_performance":   "medium",
    "tax_expenditure_report":    "medium",
    "cbk_annual_report":         "medium",
    "cbk_fsr_report":            "medium",
    "controller_of_budget":      "medium",
    "imf_report":                "medium",
    "world_bank_report":         "medium",
    "kra_corporate_plan":        "low",
    "budget_speech":             "low",
    "budget_statement":          "low",
    "post_election_report":      "medium",
    "statistical_annex":         "high",
    "revenue_grants_estimates":  "high",
    "state_corporations_annex":  "low",
    "pure_tables":               "high",
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
    "imf_report":                True,
    "world_bank_report":         True,
    "budget_speech":             False,
    "budget_statement":          True,
    "post_election_report":      True,
    "kra_corporate_plan":        False,
    "unknown":                   False,
}

SKIP_SECTIONS_MAP = {
    "budget_policy_statement":  ["foreword", "acknowledgement", "table of contents"],
    "budget_review_outlook":    ["foreword", "acknowledgement", "table of contents"],
    "budget_summary":           [],
    "controller_of_budget":     ["county breakdown", "appendix", "foreword", "acknowledgement"],
    "auditor_general_report":   ["foreword", "table of contents"],
    "cbk_annual_report": [
        "directors report", "directors' report",
        "financial statements", "statement of financial position",
        "income statement", "cash flow statement",
        "notes to financial statements", "staff costs",
        "human resources", "corporate governance", "board committees",
        "cbk balance sheet",
    ],
    "cbk_mpc_report":           ["foreword", "table of contents"],
    "cbk_fsr_report":           ["foreword", "table of contents"],
    "imf_report":               ["foreword", "acknowledgement"],
    "world_bank_report":        ["foreword", "acknowledgement"],
    "economic_survey":          ["foreword", "acknowledgement"],
    "statistical_annex":        [],
    "pure_tables":              [],
    "constitution":             [],
    "finance_act":              [],
    "finance_bill":             [],
    "kra_revenue_performance":  ["foreword", "acknowledgement"],
    "kra_corporate_plan":       ["foreword", "acknowledgement"],
    "tax_expenditure_report":   ["foreword", "acknowledgement"],
    "public_debt_report":       ["foreword", "acknowledgement"],
    "debt_management_strategy": ["foreword", "acknowledgement"],
}

# BPS types: year in filename is publication year → FY = (year-1)/year
BPS_TYPES = ["budget_policy_statement", "budget_review_outlook", "budget_summary"]

# Forward-looking: year in filename = FY start
FORWARD_LOOKING_TYPES = ["debt_management_strategy"]


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    document_type:    str = "unknown"
    fiscal_year:      str = "unknown"
    report_period:    str = "annual"
    supersedes_hint:  bool = False

    # Source tracking per field
    type_source:      str = "unknown"   # filename|cover|deep|override|unknown
    fy_source:        str = "unknown"
    period_source:    str = "unknown"

    # Review level per field: none|soft|hard
    type_review:      str = "hard"
    fy_review:        str = "hard"
    period_review:    str = "hard"

    is_scanned:       bool = False
    scanned_source:   str = "unknown"   # auto|override


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — FILENAME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    return text.lower().replace(" ", ".").replace("-", ".").replace("_", ".")


def _match_type(text: str, rules: list) -> Optional[str]:
    for pattern, doc_type in rules:
        if re.search(pattern, text):
            return doc_type
    return None


def _parse_fy_range(year1: str, year2_raw: str) -> Optional[str]:
    year2_full = year2_raw if len(year2_raw) == 4 else f"20{year2_raw}"
    if abs(int(year2_full) - int(year1)) > 2:
        return None  # reject nonsensical ranges
    return f"{year1}_{year2_full[-2:]}"


def detect_from_filename(fname: str) -> DetectionResult:
    result = DetectionResult()
    f = normalize(fname)
    raw = fname.lower()

    # Type
    doc_type = _match_type(f, DOCUMENT_TYPE_RULES)
    if doc_type:
        result.document_type = doc_type
        result.type_source   = "filename"
        result.type_review   = "none"
    else:
        result.type_review   = "hard"

    # FY — explicit range first
    match = re.search(r"(20\d{2})[-_](20\d{2}|\d{2})", raw)
    if match:
        fy = _parse_fy_range(match.group(1), match.group(2))
        if fy:
            result.fiscal_year  = fy
            result.fy_source    = "filename"
            result.fy_review    = "none"
    else:
        # Single year — infer direction from doc type
        match = re.search(r"(20\d{2})", raw)
        if match:
            year = int(match.group(1))
            dt   = result.document_type
            if dt in BPS_TYPES:
                result.fiscal_year = f"{year - 1}_{str(year)[-2:]}"
            elif dt in FORWARD_LOOKING_TYPES:
                result.fiscal_year = f"{year}_{str(year + 1)[-2:]}"
            else:
                result.fiscal_year = f"{year - 1}_{str(year)[-2:]}"
            result.fy_source   = "filename"
            result.fy_review   = "soft"   # single year = soft flag

    # Period
    if re.search(r"first.three.months|three.months", f):                 p, c = "q1",       "none"
    elif re.search(r"first.six.months|six.months|half.year|first.half", f): p, c = "h1",    "none"
    elif re.search(r"second.six.months|second.half", f):                 p, c = "h2",       "none"
    elif re.search(r"nine.months|first.nine.months", f):                 p, c = "q3",       "none"
    elif re.search(r"mid.year", f):                                      p, c = "mid_year",  "none"
    elif re.search(r"\bannual\b|\byearly\b", f):                         p, c = "annual",    "none"
    elif result.document_type == "cbk_mpc_report":
        mpc = re.search(r"(\d+)(?:st|nd|rd|th)", f)
        if mpc:
            num  = int(mpc.group(1))
            # 16=even=April=H1, 17=odd=October=H2
            p, c = ("h1" if num % 2 == 0 else "h2"), "none"
        else:
            p, c = "annual", "soft"
    else:
        p, c = "annual", "soft"

    result.report_period   = p
    result.period_source   = "filename"
    result.period_review   = c

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: Path, char_start: int = 0, char_end: int = 3000) -> tuple:
    """
    Extract a slice of text from a PDF.
    Returns (text, is_scanned).
    is_scanned=True if extracted text is under 100 chars (image-based PDF).
    """
    if not PYMUPDF_AVAILABLE:
        return "", False

    try:
        doc  = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text("text")
            if len(text) >= char_end:
                break
        doc.close()
        text      = text[char_start:char_end]
        is_scanned = len(text.strip()) < 100
        return text, is_scanned
    except Exception as e:
        print(f"    ⚠️  PDF read error: {e}")
        return "", False


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — COVER TEXT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_fy_from_text(text: str) -> Optional[str]:
    """Try all FY patterns against text. Returns fy string or None."""
    t = text.lower()
    for pattern, mode in COVER_FY_PATTERNS:
        m = re.search(pattern, t)
        if not m:
            continue
        if mode == "range":
            try:
                y1 = m.group(1)
                y2 = m.group(2)
                fy = _parse_fy_range(y1, y2)
                if fy:
                    return fy
            except IndexError:
                pass
        elif mode == "end_year":
            year = int(m.group(1))
            return f"{year - 1}_{str(year)[-2:]}"
    return None


def _detect_period_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for pattern, period in COVER_PERIOD_PATTERNS:
        if re.search(pattern, t):
            return period
    return None


def _detect_supersedes_hint(text: str) -> bool:
    t = text.lower()
    for pattern in SUPERSEDES_HINT_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


def detect_from_cover(text: str, current: DetectionResult) -> DetectionResult:
    """
    Update result fields using cover text.
    Filename wins for document_type — cover only fills gaps.
    Cover wins for FY and period over filename single-year inference.
    """
    result = DetectionResult(
        document_type   = current.document_type,
        fiscal_year     = current.fiscal_year,
        report_period   = current.report_period,
        supersedes_hint = current.supersedes_hint,
        type_source     = current.type_source,
        fy_source       = current.fy_source,
        period_source   = current.period_source,
        type_review     = current.type_review,
        fy_review       = current.fy_review,
        period_review   = current.period_review,
        is_scanned      = current.is_scanned,
        scanned_source  = current.scanned_source,
    )

    t = normalize(text)

    # Type — only fill if filename gave unknown
    if result.document_type == "unknown":
        doc_type = _match_type(t, COVER_TYPE_RULES)
        if doc_type:
            result.document_type = doc_type
            result.type_source   = "cover"
            result.type_review   = "soft"  # cover type = soft, not hard

    # FY — cover wins over soft filename inference
    if result.fy_review in ("hard", "soft"):
        fy = _detect_fy_from_text(text)
        if fy:
            result.fiscal_year = fy
            result.fy_source   = "cover"
            result.fy_review   = "none"

    # Period — cover wins over soft filename inference
    if result.period_review in ("hard", "soft"):
        period = _detect_period_from_text(text)
        if period:
            result.report_period   = period
            result.period_source   = "cover"
            result.period_review   = "none"

    # Supersedes hint — soft only, never treated as resolved
    if _detect_supersedes_hint(text):
        result.supersedes_hint = True

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — DEEP TEXT (FY only, targeted)
# ══════════════════════════════════════════════════════════════════════════════

def detect_from_deep_text(text: str, current: DetectionResult) -> DetectionResult:
    """Only attempts FY detection — triggered when FY still unknown after cover."""
    result = DetectionResult(
        document_type   = current.document_type,
        fiscal_year     = current.fiscal_year,
        report_period   = current.report_period,
        supersedes_hint = current.supersedes_hint,
        type_source     = current.type_source,
        fy_source       = current.fy_source,
        period_source   = current.period_source,
        type_review     = current.type_review,
        fy_review       = current.fy_review,
        period_review   = current.period_review,
        is_scanned      = current.is_scanned,
        scanned_source  = current.scanned_source,
    )

    if result.fy_review == "hard" or result.fiscal_year == "unknown":
        fy = _detect_fy_from_text(text)
        if fy:
            result.fiscal_year = fy
            result.fy_source   = "deep"
            result.fy_review   = "soft"  # deep text = soft, not none

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FINAL — APPLY OVERRIDES
# ══════════════════════════════════════════════════════════════════════════════

def apply_overrides(result: DetectionResult, fname: str) -> DetectionResult:
    if fname not in MANUAL_OVERRIDES:
        return result

    overrides = MANUAL_OVERRIDES[fname]

    if "document_type" in overrides:
        result.document_type = overrides["document_type"]
        result.type_source   = "override"
        result.type_review   = "none"

    if "fiscal_year" in overrides:
        result.fiscal_year   = overrides["fiscal_year"]
        result.fy_source     = "override"
        result.fy_review     = "none"

    if "report_period" in overrides:
        result.report_period  = overrides["report_period"]
        result.period_source  = "override"
        result.period_review  = "none"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW LEVEL ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_review_level(result: DetectionResult) -> str:
    """
    hard  — any field is still hard-unresolved
    soft  — all fields resolved but some via cover/deep/single-year inference
    none  — all fields resolved with high confidence
    "na" fiscal year (e.g. constitution) is intentional — does not trigger hard
    """
    reviews = [result.type_review, result.period_review]
    # Only include fy_review if FY is not intentionally "na"
    if result.fiscal_year != "na":
        reviews.append(result.fy_review)
    if "hard" in reviews:
        return "hard"
    if "soft" in reviews:
        return "soft"
    return "none"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def process_document(pdf_path: Path, use_pdf: bool) -> DetectionResult:
    fname = pdf_path.name

    # Pass 1 — filename
    result = detect_from_filename(fname)
    result = apply_overrides(result, fname)

    if not use_pdf or not PYMUPDF_AVAILABLE:
        return result

    needs_cover = (
        result.type_review   != "none" or
        result.fy_review     != "none" or
        result.period_review != "none"
    )

    if not needs_cover:
        return result

    # Pass 2 — cover text
    cover_text, is_scanned = extract_pdf_text(pdf_path, 0, 3000)

    if is_scanned:
        result.is_scanned     = True
        result.scanned_source = "auto"
        # Scanned — can't read text, mark remaining unknowns as hard
        return apply_overrides(result, fname)

    result = detect_from_cover(cover_text, result)
    result = apply_overrides(result, fname)

    # Pass 3 — deep text, only if FY still unresolved (skip if intentionally "na")
    if result.fiscal_year not in ("na",) and (result.fy_review == "hard" or result.fiscal_year == "unknown"):
        deep_text, _ = extract_pdf_text(pdf_path, 3000, 10000)
        if deep_text:
            result = detect_from_deep_text(deep_text, result)
            result = apply_overrides(result, fname)

    return result


def generate_config(raw_dir: Path, use_pdf: bool) -> dict:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    print(f"Found {len(pdfs)} PDF files")
    if use_pdf and PYMUPDF_AVAILABLE:
        print("PDF cover text detection: ENABLED\n")
    else:
        print("PDF cover text detection: DISABLED (filename only)\n")

    documents    = []
    counts       = {"none": 0, "soft": 0, "hard": 0}

    for pdf in pdfs:
        fname  = pdf.name
        result = process_document(pdf, use_pdf)
        level  = assign_review_level(result)
        counts[level] += 1

        priority   = PRIORITY_MAP.get(result.document_type, "low")
        category   = CATEGORY_MAP.get(result.document_type, 0)
        rag_weight = RAG_WEIGHT_MAP.get(priority, 1.0)

        entry = {
            "_review_level":      level,
            "_supersedes_review": result.document_type == "budget_review_outlook",
            "_supersedes_hint":   result.supersedes_hint,
            "filename":           fname,
            "document_type":      result.document_type,
            "fiscal_year":        result.fiscal_year,
            "report_period":      result.report_period,
            "domain":             DOMAIN_MAP.get(result.document_type, "unknown"),
            "priority":           priority,
            "rag_weight":         rag_weight,
            "category":           category,
            "chunking_strategy":  CHUNKING_STRATEGY_MAP.get(category, "narrative"),
            "chunk_by":           CHUNK_BY_MAP.get(category, "token"),
            "chunk_size":         CHUNK_SIZE_MAP.get(category, 350),
            "chunk_overlap":      CHUNK_OVERLAP_MAP.get(category, 50),
            "index":              True,
            "has_tables":         HAS_TABLES_MAP.get(result.document_type, False),
            "is_scanned":         result.is_scanned,
            "language":           "english",
            "skip_sections":      SKIP_SECTIONS_MAP.get(result.document_type, []),
            "supersedes":         None,
            "superseded_by":      None,
            # Audit trail
            "type_source":        result.type_source,
            "fy_source":          result.fy_source,
            "period_source":      result.period_source,
        }

        documents.append(entry)

        icon = "✓" if level == "none" else ("～" if level == "soft" else "⚠️ ")
        print(
            f"  {icon:3} [{level:<4}] "
            f"{fname[:52]:<52} | "
            f"{result.document_type:<30} | "
            f"{result.fiscal_year:<10} | "
            f"{result.report_period}"
        )

    config = {
        "pipeline": {
            "agent":             "finance",
            "chunk_min_tokens":  CHUNK_MIN_TOKENS,
            "chunk_max_tokens":  CHUNK_MAX_TOKENS,
            "dedup_threshold":   0.85,
            "qdrant_collection": "kenya_executive_roundtable",
        },
        "documents": documents,
    }

    print(f"\n{'='*70}")
    print(f"Total     : {len(documents)}")
    print(f"✓  none   : {counts['none']}  (fully resolved)")
    print(f"～  soft   : {counts['soft']}  (spot-check recommended)")
    print(f"⚠️  hard   : {counts['hard']}  (manual fix required)")
    print(f"{'='*70}\n")

    return config


# ══════════════════════════════════════════════════════════════════════════════
# YAML WRITER
# ══════════════════════════════════════════════════════════════════════════════

def safe_yaml_str(value) -> str:
    if value is None:                   return "null"
    if isinstance(value, bool):         return str(value).lower()
    if isinstance(value, (int, float)): return str(value)
    if isinstance(value, str):
        # Quote if contains special chars that would break YAML, otherwise bare
        if any(c in value for c in ":#{}[]|>&*!,"):
            return f"'{value}'"
        return value
    dumped = yaml.dump(value, default_flow_style=True, allow_unicode=True)
    return dumped.strip()


def write_yaml(config: dict, output_path: Path):
    lines = []
    lines.append("# Kenya AI Executive Roundtable — RAG Pipeline Config")
    lines.append("# Auto-generated by generate_config.py — RAG only")
    lines.append("# Review levels: none=resolved | soft=spot-check | hard=manual fix required")
    lines.append("# type_source / fy_source / period_source show how each field was resolved")
    lines.append("# Search 'HARD' to find fields needing manual fixes")
    lines.append("# Search 'SOFT' to find fields worth spot-checking")
    lines.append("")

    lines.append("pipeline:")
    for k, v in config["pipeline"].items():
        lines.append(f"  {k}: {safe_yaml_str(v)}")
    lines.append("")

    lines.append("documents:")

    for doc in config["documents"]:
        level             = doc.pop("_review_level",      "hard")
        supersedes_review = doc.pop("_supersedes_review", False)
        supersedes_hint   = doc.pop("_supersedes_hint",   False)

        lines.append("")
        if level == "hard":
            lines.append("  # ⚠️  HARD — manual fix required")
        elif level == "soft":
            lines.append("  # ～  SOFT — spot-check recommended")

        lines.append(f"  - filename: {safe_yaml_str(doc['filename'])}")

        type_flag = "  # HARD — unknown type, fix manually" if doc['document_type'] == 'unknown' else ""
        lines.append(f"    document_type: {safe_yaml_str(doc['document_type'])}{type_flag}")

        fy_flag = ""
        if doc['fiscal_year'] == 'na':
            fy_flag = "  # timeless — no fiscal year applies"
        elif doc['fiscal_year'] == 'unknown':
            fy_flag = "  # HARD — could not detect fiscal year"
        elif doc['fy_source'] in ('cover', 'deep'):
            fy_flag = "  # SOFT — resolved from document text"
        elif doc['fy_source'] == 'filename' and level == 'soft':
            fy_flag = "  # SOFT — inferred from single year in filename"
        lines.append(f"    fiscal_year: {safe_yaml_str(doc['fiscal_year'])}{fy_flag}")

        period_flag = ""
        if doc['period_source'] in ('cover',):
            period_flag = "  # SOFT — resolved from document text"
        lines.append(f"    report_period: {safe_yaml_str(doc['report_period'])}{period_flag}")

        lines.append(f"    domain: {safe_yaml_str(doc['domain'])}")
        lines.append(f"    priority: {safe_yaml_str(doc['priority'])}")
        lines.append(f"    rag_weight: {doc['rag_weight']}")
        lines.append(f"    category: {doc['category']}")
        lines.append(f"    chunking_strategy: {safe_yaml_str(doc['chunking_strategy'])}")
        lines.append(f"    chunk_by: {safe_yaml_str(doc['chunk_by'])}")
        lines.append(f"    chunk_size: {doc['chunk_size']}")
        lines.append(f"    chunk_overlap: {doc['chunk_overlap']}")
        lines.append(f"    index: {safe_yaml_str(doc['index'])}")
        lines.append(f"    has_tables: {safe_yaml_str(doc['has_tables'])}")
        lines.append(f"    is_scanned: {safe_yaml_str(doc['is_scanned'])}")
        lines.append(f"    language: {safe_yaml_str(doc['language'])}")

        if doc['skip_sections']:
            lines.append(f"    skip_sections:")
            for s in doc['skip_sections']:
                lines.append(f"      - {safe_yaml_str(s)}")
        else:
            lines.append(f"    skip_sections: []")

        # Supersedes
        if supersedes_review and supersedes_hint:
            lines.append(f"    supersedes: null  # SOFT — doc hints at revision, verify which BPS figures")
        elif supersedes_review:
            lines.append(f"    supersedes: null  # HARD — BROP doc, specify which BPS figures this revises")
        else:
            lines.append(f"    supersedes: null")

        lines.append(f"    superseded_by: null")

        # Audit trail
        lines.append(f"    # resolved: type={doc['type_source']} fy={doc['fy_source']} period={doc['period_source']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Config written → {output_path}")
    print(f"   Search 'HARD' for manual fixes, 'SOFT' for spot-checks.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate RAG pipeline config.yaml with 3-pass detection"
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Path to raw PDFs directory (default: data/raw)"
    )
    parser.add_argument(
        "--output", type=str, default="config.yaml",
        help="Output path for config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF cover text detection — filename only (faster)"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raw_dir = Path(__file__).parent / args.raw_dir
    if not raw_dir.exists():
        print(f"❌ Directory not found: {raw_dir}")
        return

    output_path = Path(args.output)
    use_pdf     = not args.no_pdf

    print(f"Scanning : {raw_dir}")
    print(f"Output   : {output_path}\n")

    config = generate_config(raw_dir, use_pdf)
    write_yaml(config, output_path)


if __name__ == "__main__":
    main()