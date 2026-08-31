#!/usr/bin/env python3
"""Create appendix tables for the error-attribution human audit.

The primary validation sample uses one assigned annotation for each of the 53
unique divergences (Ben's package A and David's package B). Johanna independently
annotated package B, providing a 27-divergence inter-annotator subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


QUESTIONS = {
    "q_real": "Meaningful divergence",
    "q_source": "Error source",
    "q_type": "Divergence type",
    "q_severity": "Severity",
}

RESPONSE_CATEGORIES = {
    "q_real": ("yes", "unsure", "no"),
    "q_source": ("correct", "unsure", "incorrect"),
    "q_type": ("correct", "unsure", "incorrect"),
    "q_severity": ("correct", "unsure", "incorrect"),
}

COARSE_SOURCE = {
    "Agent contradicted summary": "Agent error",
    "Agent missed": "Agent error",
    "Summary gap (contradicts)": "Extractor error",
    "Summary omission": "Extractor error",
    "Paper underspecified": "Original error",
    "Paper-code mismatch": "Original error",
    "Insufficient specification": "Original error",
    "Data not in package": "Data missing",
    "Unexplained": "Other / unknown",
}

SOURCE_ORDER = [
    "Agent error",
    "Original error",
    "Extractor error",
    "Data missing",
    "Other / unknown",
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=repo / "human_audit")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "analysis_output_regrade_na" / "human_audit",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires two non-empty, equally sized vectors")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    categories = set(left) | set(right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[value] / len(left) * right_counts[value] / len(right)
        for value in categories
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def gwet_ac1(
    left: list[str], right: list[str], categories: tuple[str, ...]
) -> float | None:
    """Calculate nominal Gwet's AC1 for two raters.

    AC1 uses an expected-agreement term that remains informative when ratings
    are concentrated in one category. ``categories`` is the full response
    scale, including categories not observed in this particular subset.
    """
    if len(left) != len(right) or not left:
        raise ValueError("AC1 requires two non-empty, equally sized vectors")
    unknown = (set(left) | set(right)) - set(categories)
    if unknown:
        raise ValueError(f"AC1 received unknown response categories: {unknown}")

    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    pooled = Counter(left + right)
    marginals = {
        value: pooled[value] / (2 * len(left))
        for value in categories
    }
    expected = sum(p * (1.0 - p) for p in marginals.values()) / (
        len(categories) - 1
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def count_row(label: str, rows: list[dict[str, str]], question: str) -> dict:
    values = Counter(row[question] for row in rows)
    if question == "q_real":
        positive, negative = values["yes"], values["no"]
    else:
        positive, negative = values["correct"], values["incorrect"]
    return {
        "target": label,
        "positive": positive,
        "unsure": values["unsure"],
        "negative": negative,
        "n": len(rows),
    }


def cell(count: int, denominator: int) -> str:
    if denominator == 0:
        return "--"
    return f"{count} ({100 * count / denominator:.0f}\\%)"


def validation_tex(summary: list[dict], sources: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Human validation of the error-attribution pipeline.}",
        r"\label{tab:human-audit-validation}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Audit target & Correct/yes & Unsure & Incorrect/no & $N$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel A: Validation outcomes}} \\",
    ]
    for row in summary:
        lines.append(
            f"{row['target']} & {cell(row['positive'], row['n'])} & "
            f"{cell(row['unsure'], row['n'])} & {cell(row['negative'], row['n'])} "
            f"& {row['n']} \\\\"
        )
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel B: Error-source validation by assigned source}} \\",
    ])
    for row in sources:
        lines.append(
            f"{row['target']} & {cell(row['positive'], row['n'])} & "
            f"{cell(row['unsure'], row['n'])} & {cell(row['negative'], row['n'])} "
            f"& {row['n']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.92\textwidth}",
        r"\footnotesize \textit{Notes:} The primary sample contains 53 divergences from 27 table-runs spanning six tables in four purposively selected short papers. Panel A reports one assigned annotation per divergence. Error-source, divergence-type, and severity validation are conditional on the annotator confirming a meaningful divergence ($N=49$). Panel B maps the detailed pipeline labels to the five error-source groups reported in the main paper. The audit sample contains no data-missing cases. Percentages may not sum to 100 due to rounding.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def agreement_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Inter-annotator agreement in the double-coded audit subset.}",
        r"\label{tab:human-audit-agreement}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Audit target & Exact agreement & Cohen's $\kappa$ & Gwet's AC1 & $N$ \\",
        r"\midrule",
    ]
    for row in rows:
        kappa_text = "--" if row["kappa"] is None else f"{row['kappa']:.2f}"
        lines.append(
            f"{row['target']} & {row['agreements']} ({100 * row['agreement']:.0f}\\%) "
            f"& {kappa_text} & {row['gwet_ac1']:.2f} & {row['n']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.92\textwidth}",
        r"\footnotesize \textit{Notes:} Two annotators independently rated the same 27 divergences from the runs from two papers. We treat the three responses for each target as nominal categories and report exact agreement as the primary measure. Cohen's kappa is undefined for meaningful-divergence detection because both annotators answered yes in every case; for divergence type, one annotator answered correct in all 27 cases while the other answered correct in 26 and unsure in one, yielding 96\% exact agreement but $\kappa=0$. Gwet's AC1 is less sensitive to such concentrated marginal distributions. Error-source disagreements mostly reflect different use of the ``unsure'' category.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def markdown_table(summary: list[dict], sources: list[dict], agreement: list[dict]) -> str:
    lines = [
        "# Human-audit appendix tables",
        "",
        "## Validation outcomes",
        "",
        "| Audit target | Correct/yes | Unsure | Incorrect/no | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target']} | {row['positive']} ({100*row['positive']/row['n']:.1f}%) "
            f"| {row['unsure']} ({100*row['unsure']/row['n']:.1f}%) "
            f"| {row['negative']} ({100*row['negative']/row['n']:.1f}%) | {row['n']} |"
        )
    lines.extend([
        "",
        "## Error-source validation by main-paper category",
        "",
        "| Assigned source | Correct | Unsure | Incorrect | N |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in sources:
        if row["n"]:
            values = [f"{row[key]} ({100*row[key]/row['n']:.1f}%)" for key in ("positive", "unsure", "negative")]
        else:
            values = ["--", "--", "--"]
        lines.append(f"| {row['target']} | {values[0]} | {values[1]} | {values[2]} | {row['n']} |")
    lines.extend([
        "",
        "## Inter-annotator agreement",
        "",
        "| Audit target | Exact agreement | Cohen's kappa | Gwet's AC1 | N |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in agreement:
        kappa_text = "undefined" if row["kappa"] is None else f"{row['kappa']:.3f}"
        lines.append(
            f"| {row['target']} | {row['agreements']}/{row['n']} "
            f"({100*row['agreement']:.1f}%) | {kappa_text} "
            f"| {row['gwet_ac1']:.3f} | {row['n']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ben = read_csv(args.audit_dir / "audit_annotations_Ben.csv")
    david = read_csv(args.audit_dir / "audit_annotations_David.csv")
    primary = ben + david
    if len(primary) != 53 or len({row["audit_id"] for row in primary}) != 53:
        raise ValueError("Expected 53 unique primary audit records")

    johanna_export = json.loads(
        (args.audit_dir / "audit_annotations_Johanna.json").read_text(encoding="utf-8")
    )
    johanna_annotations = johanna_export["annotations"]
    david_by_id = {row["audit_id"]: row for row in david}
    if set(johanna_annotations) != set(david_by_id):
        raise ValueError("Johanna and David must have rated the same audit IDs")

    johanna = []
    for audit_id, annotation in johanna_annotations.items():
        row = dict(david_by_id[audit_id])
        row.update({
            "annotator": johanna_export.get("annotator", "Johanna"),
            "q_real": annotation.get("real", ""),
            "q_source": annotation.get("source", ""),
            "q_type": annotation.get("type", ""),
            "q_severity": annotation.get("severity", ""),
            "notes": annotation.get("notes", ""),
            "ts": annotation.get("ts", ""),
        })
        johanna.append(row)

    confirmed = [row for row in primary if row["q_real"] == "yes"]
    summary = [count_row(QUESTIONS["q_real"], primary, "q_real")]
    summary.extend(
        count_row(QUESTIONS[question], confirmed, question)
        for question in ("q_source", "q_type", "q_severity")
    )

    sources = []
    for source in SOURCE_ORDER:
        source_rows = [
            row for row in confirmed
            if COARSE_SOURCE.get(row["root_cause"], "Other / unknown") == source
        ]
        sources.append(count_row(source, source_rows, "q_source"))

    agreement = []
    johanna_by_id = {row["audit_id"]: row for row in johanna}
    for question, label in QUESTIONS.items():
        left = [david_by_id[audit_id][question] for audit_id in sorted(david_by_id)]
        right = [johanna_by_id[audit_id][question] for audit_id in sorted(david_by_id)]
        agreements = sum(a == b for a, b in zip(left, right))
        agreement.append({
            "target": label,
            "agreements": agreements,
            "agreement": agreements / len(left),
            "kappa": kappa(left, right),
            "gwet_ac1": gwet_ac1(left, right, RESPONSE_CATEGORIES[question]),
            "n": len(left),
        })

    all_rows = primary + johanna
    write_csv(
        args.output_dir / "human_audit_annotations_clean.csv",
        all_rows,
        list(all_rows[0]),
    )
    write_csv(
        args.output_dir / "human_audit_validation.csv",
        summary,
        ["target", "positive", "unsure", "negative", "n"],
    )
    write_csv(
        args.output_dir / "human_audit_by_error_source.csv",
        sources,
        ["target", "positive", "unsure", "negative", "n"],
    )
    write_csv(
        args.output_dir / "human_audit_agreement.csv",
        agreement,
        ["target", "agreements", "agreement", "kappa", "gwet_ac1", "n"],
    )
    (args.output_dir / "human_audit_validation.tex").write_text(
        validation_tex(summary, sources), encoding="utf-8"
    )
    (args.output_dir / "human_audit_agreement.tex").write_text(
        agreement_tex(agreement), encoding="utf-8"
    )
    (args.output_dir / "human_audit_tables.md").write_text(
        markdown_table(summary, sources, agreement), encoding="utf-8"
    )
    (args.output_dir / "human_audit_summary.json").write_text(
        json.dumps({
            "design": {
                "unique_divergences": len(primary),
                "confirmed_divergences": len(confirmed),
                "papers": len({row["paper_slug"] for row in primary}),
                "tables": len({(row["paper_slug"], row["output"]) for row in primary}),
                "table_runs": len({
                    (row["paper_slug"], row["agent_label"], row["output"])
                    for row in primary
                }),
                "double_coded_divergences": len(david),
                "total_ratings": len(all_rows),
            },
            "validation": summary,
            "by_error_source": sources,
            "agreement": agreement,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote human-audit appendix tables to {args.output_dir}")


if __name__ == "__main__":
    main()
