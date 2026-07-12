# NISM Series X-A: Investment Adviser (Level 1) — Exam Blueprint

> Research date: 2026-07-13. All facts below are sourced from the official NISM
> website (`nism.ac.in`). Source URLs are listed in the **SOURCES** section and
> cited inline as `[S#]`.

---

## ⚠️ SPEC VERIFICATION SUMMARY (read first)

The project spec (`atlas-spec/017`) claims:

> "90 standalone (1 mark) + 6 caselets×5 (1 mark) + 3 caselets×5 (2 mark)
> = 135 Q, 150 marks, 180 min, pass 90/150, negative 25% of the question's marks."

**RESULT: EVERY field in the spec is CONFIRMED correct against the official
NISM assessment structure `[S2]`. No corrections required.**

The only nuance worth noting: NISM's own summary sentence phrases it as
"90 independent multiple-choice questions **and 9 caselets/case-based questions**".
The "9 caselets" expand to 6+3 caselets × 5 questions each = 45 case-based
questions, so 90 + 45 = **135 total questions** — exactly matching the spec's
arithmetic.

---

## VERIFIED EXAM PATTERN

Official source text `[S2]`:

> "The Investment Adviser certification examination consists of 90 independent
> multiple-choice questions and 9 caselets/case-based questions and should be
> completed in 3 hours. The passing score for the examination is 60% which is
> 90 marks out of total 150 marks. There shall be negative marking of 25% of
> the marks assigned to a question for each wrong answer."
>
> Multiple Choice Questions [90 questions of 1 mark each] → 90 marks
> 9 Case-based Questions:
>   [6 caselets (each case with 5 questions of 1 mark each)] → 6×5×1 = 30 marks
>   [3 caselets (with 5 questions of 2 marks each)]         → 3×5×2 = 30 marks
> Total → 150 marks

Confirmed also by the exam quick-facts table on the detail page `[S1]`:
Fees ₹3000+GST · Test Duration **180** min · No. of Questions **135** ·
Maximum Marks **150** · Pass Marks **60%** · Certificate Validity 3 years.

| Field | Spec claim | Official value | Match | Citation |
|---|---|---|---|---|
| Total questions | 135 | 135 (90 standalone + 45 case-based) | ✓ | `[S1]` `[S2]` |
| Standalone MCQs | 90 × 1 mark | 90 × 1 mark = 90 marks | ✓ | `[S2]` |
| 1-mark caselets | 6 caselets × 5 Q × 1 mark | 6×5×1 = 30 marks | ✓ | `[S2]` |
| 2-mark caselets | 3 caselets × 5 Q × 2 marks | 3×5×2 = 30 marks | ✓ | `[S2]` |
| Total marks | 150 | 150 | ✓ | `[S1]` `[S2]` |
| Duration | 180 min | 180 min (3 hours) | ✓ | `[S1]` `[S2]` |
| Passing score | 90 / 150 | 90 / 150 = 60% | ✓ | `[S1]` `[S2]` |
| Negative marking | 25% of question's marks | 25% of marks assigned to the question | ✓ | `[S2]` |
| Uses caselets? | yes, 9 | yes — 9 caselets / case-based question sets | ✓ | `[S2]` |

Every field: **✓ CONFIRMED**. No ✗, no mismatch.

---

## CHAPTERS

The official Test Objectives page `[S3]` groups the workbook into **6 modules /
20 chapters**. Ordered list (titles verbatim from the official outline):

**MODULE 1: PERSONAL FINANCIAL PLANNING**
1. Introduction to Personal Financial Planning
2. Time Value of Money
3. Cash Flow Management and Budgeting
4. Debt Management and Loans

**MODULE 2: INDIAN FINANCIAL MARKETS**
5. Introduction to Indian Financial Markets
6. Securities Market Segments

**MODULE 3: INVESTMENT PRODUCTS**
7. Introduction to Investments
8. Investing in Stocks
9. Investing in Fixed Income Securities
10. Understanding Derivatives

**MODULE 4: INVESTMENT THROUGH MANAGED PORTFOLIO**
11. Mutual Fund
12. Portfolio Manager
13. Overview of Alternative Investment Funds (AIFs)

**MODULE 5: PORTFOLIO CONSTRUCTION, PERFORMANCE MONITORING AND EVALUATION**
14. Introduction to Modern Portfolio Theory
15. Portfolio Construction Process
16. Portfolio Performance Measurement and Evaluation

**MODULE 6: OPERATIONS, REGULATORY ENVIRONMENT, COMPLIANCE AND ETHICS**
17. Operational Aspects of Investment Management
18. Key Regulations
19. Ethical Issues
20. Grievance Redress Mechanism

Chapter count: **20** (across 6 modules).

---

## BLUEPRINT WEIGHTAGE

**Official per-chapter / per-module weightage: NOT PUBLISHED.**

NISM's Test Objectives page `[S3]` lists the modules, chapters, and detailed
sub-topics (learning objectives), but publishes **no marks or percentage
weight per unit** for NISM-Series-X-A. The detail page `[S1]` and assessment
structure `[S2]` likewise give only the overall pattern, not unit weights.

Therefore the block below is a **FALLBACK: even split** across all 20 chapters
(0.05 each), summing to exactly 1.0. It is **UNOFFICIAL** — NISM has not
published unit weightage for this exam. If a weighted distribution is desired,
it must be derived from workbook page-count/topic-depth heuristics or from
observed past-paper frequency, and should still be labeled unofficial.

```yaml
# blueprint.overrides — NISM Series X-A (Investment Adviser Level 1)
# Source of chapter list: official NISM Test Objectives [S3]
# Weightage status: FALLBACK (even split) — NISM does NOT publish unit weightage.
# 20 chapters × 0.05 = 1.0
blueprint:
  overrides:
    ch01_intro_personal_financial_planning: 0.05   # FALLBACK
    ch02_time_value_of_money: 0.05                  # FALLBACK
    ch03_cash_flow_management_budgeting: 0.05       # FALLBACK
    ch04_debt_management_loans: 0.05                # FALLBACK
    ch05_intro_indian_financial_markets: 0.05       # FALLBACK
    ch06_securities_market_segments: 0.05           # FALLBACK
    ch07_intro_investments: 0.05                    # FALLBACK
    ch08_investing_in_stocks: 0.05                  # FALLBACK
    ch09_investing_fixed_income_securities: 0.05    # FALLBACK
    ch10_understanding_derivatives: 0.05            # FALLBACK
    ch11_mutual_fund: 0.05                          # FALLBACK
    ch12_portfolio_manager: 0.05                    # FALLBACK
    ch13_overview_alternative_investment_funds: 0.05 # FALLBACK
    ch14_intro_modern_portfolio_theory: 0.05        # FALLBACK
    ch15_portfolio_construction_process: 0.05       # FALLBACK
    ch16_portfolio_performance_measurement: 0.05    # FALLBACK
    ch17_operational_aspects_investment_mgmt: 0.05  # FALLBACK
    ch18_key_regulations: 0.05                      # FALLBACK
    ch19_ethical_issues: 0.05                       # FALLBACK
    ch20_grievance_redress_mechanism: 0.05          # FALLBACK
# sum = 1.00
```

### Optional module-level fallback (6 modules, chapter-count weighted)

If you prefer to weight by module rather than chapter, an even-per-chapter
roll-up gives (still UNOFFICIAL):

| Module | Chapters | Weight |
|---|---|---|
| M1 Personal Financial Planning | 4 | 0.20 |
| M2 Indian Financial Markets | 2 | 0.10 |
| M3 Investment Products | 4 | 0.20 |
| M4 Investment through Managed Portfolio | 3 | 0.15 |
| M5 Portfolio Construction, Perf. Monitoring & Evaluation | 3 | 0.15 |
| M6 Operations, Regulatory Env., Compliance & Ethics | 4 | 0.20 |
| **Total** | **20** | **1.00** |

---

## SOURCES

- `[S1]` NISM — Investment Adviser (Level 1) certification detail page (exam
  quick-facts table: fees, 180 min, 135 questions, 150 marks, 60% pass, 3-yr
  validity; examination objectives; assessment structure):
  https://www.nism.ac.in/investment-adviser-level-1/
- `[S2]` NISM — Assessment Structure text for NISM-Series-X-A (same page as
  `[S1]`; quoted verbatim above): https://www.nism.ac.in/investment-adviser-level-1/
- `[S3]` NISM — Test Objectives, NISM-Series-X-A Investment Adviser (Level 1)
  (official module + chapter + sub-topic outline; no unit weightage published):
  https://www.nism.ac.in/test-objectives-investment-adviser-level-1/
- `[S4]` NISM — Certification examinations index (confirms exam name/series ID):
  https://www.nism.ac.in/investment-adviser-level-1-certification-examination/

All sources are official (`nism.ac.in`). No third-party / coaching sites were
used for the exam pattern, chapter list, or the (absent) weightage.
