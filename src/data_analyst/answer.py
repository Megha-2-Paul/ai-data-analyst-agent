"""Deterministic evidence extraction and analyst-style answer generation.

Stage 3 keeps answer generation grounded in structured analytical results. It does
not ask an LLM to invent prose or statistics: every finding is created from
values already present in the analytical result and carries evidence IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class AnswerGenerationError(ValueError):
    """Raised when an answer cannot be grounded in the supplied evidence."""


@dataclass(frozen=True)
class EvidenceItem:
    """One source-backed fact that may support an analyst finding."""

    evidence_id: str
    source_step_id: str
    claim_type: str
    claim: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    """A deterministic conclusion linked to one or more evidence items."""

    finding_id: str
    text: str
    evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalystAnswer:
    """Human-readable answer plus the evidence and limitations behind it."""

    summary: str
    findings: list[Finding]
    evidence: list[EvidenceItem]
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": [item.to_dict() for item in self.findings],
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": list(self.limitations),
        }

    def render_text(self) -> str:
        lines = [self.summary]
        if self.findings:
            lines.append("")
            lines.append("Findings:")
            lines.extend(f"- {finding.text}" for finding in self.findings)
        if self.limitations:
            lines.append("")
            lines.append("Limitations:")
            lines.extend(f"- {item}" for item in self.limitations)
        return "\n".join(lines)


def _rows(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    rows = result.get("result")
    return rows if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _add(
    evidence: list[EvidenceItem],
    source: str,
    claim_type: str,
    claim: str,
    **values: Any,
) -> str:
    evidence_id = f"e{len(evidence) + 1}"
    evidence.append(EvidenceItem(evidence_id, source, claim_type, claim, values))
    return evidence_id


def _single_step_evidence(result: dict[str, Any], source: str = "step_1") -> tuple[list[EvidenceItem], list[Finding], str, list[str]]:
    analysis = result.get("analysis")
    evidence: list[EvidenceItem] = []
    findings: list[Finding] = []
    limitations: list[str] = []

    if analysis in {"grouped_aggregation", "aggregation"}:
        rows = _rows(result)
        parameters = result.get("parameters", {})
        metric = parameters.get("metric")
        group_columns = parameters.get("group_by", [])
        if not rows:
            return [], [], "No results were returned for the requested aggregation.", ["The analytical result contained no rows."]
        if not metric or not group_columns:
            return [], [], "The grouped analysis completed.", ["The grouped result did not include a metric and grouping column."]
        
        metric_values = [
            (row.get(metric), row)
            for row in rows
            if metric in row and row.get(metric) is not None
        ]
        if not metric_values:
            return [], [], "The grouped analysis completed.", ["No non-null metric values were available."]
        
        # Respect the question's intent. A plain "by" question asks for the
        # grouped breakdown; only explicit ranking language asks us to select
        # one group.
        question = str(result.get("question", "")).lower()
        sort_direction = parameters.get("sort_direction")
        if sort_direction == "desc" or any(term in question for term in ("highest", "largest", "maximum", "top")):
            selected = [max(metric_values, key=lambda item: item[0])]
            claim_type = "highest_group"
        elif sort_direction == "asc" or any(term in question for term in ("lowest", "smallest", "minimum", "bottom")):
            selected = [min(metric_values, key=lambda item: item[0])]
            claim_type = "lowest_group"
        else:
            selected = metric_values
            claim_type = "grouped_value"
        
        evidence_ids: list[str] = []
        value_parts: list[str] = []
        for value, row in selected:
            context = ", ".join(f"{column}={row.get(column)}" for column in group_columns)
            eid = _add(
                evidence,
                source,
                claim_type,
                f"{metric} for {context}",
                metric=metric,
                value=value,
                context=context,
            )
            evidence_ids.append(eid)
            value_parts.append(f"{context}: {_fmt(value)}")
        
        if claim_type == "highest_group":
            context = ", ".join(f"{column}={selected[0][1].get(column)}" for column in group_columns)
            text = f"The highest {metric} is {_fmt(selected[0][0])} for {context}."
            summary = text
        elif claim_type == "lowest_group":
            context = ", ".join(f"{column}={selected[0][1].get(column)}" for column in group_columns)
            text = f"The lowest {metric} is {_fmt(selected[0][0])} for {context}."
            summary = text
        else:
            text = f"{metric} by {', '.join(group_columns)}: " + "; ".join(value_parts) + "."
            summary = f"The grouped analysis returned {len(selected)} {('group' if len(selected) == 1 else 'groups')} for {metric}."
        
        findings.append(Finding("f1", text, evidence_ids))
        return evidence, findings, summary, limitations

    if analysis == "descriptive_statistics":
        rows = _rows(result)
        for idx, row in enumerate(rows, start=1):
            column = row.get("column", "value")
            if row.get("mean") is not None:
                eid = _add(evidence, source, "descriptive_mean", f"Mean of {column}", column=column, value=row["mean"])
                findings.append(Finding(f"f{idx}", f"The mean {column} is {_fmt(row['mean'])}.", [eid]))
        if not findings:
            limitations.append("No descriptive statistics with non-null means were returned.")
        return evidence, findings, "Descriptive statistics were calculated from the dataset.", limitations

    if analysis == "correlation":
        rows = _rows(result)
        pairs = []
        for row in rows:
            names = list(row.keys())
            if len(names) >= 2:
                target = names[0]
                for other in names[1:]:
                    value = row.get(other)
                    if isinstance(value, (int, float)) and other != target:
                        pairs.append((abs(value), target, other, value))
        if pairs:
            _, left, right, value = max(pairs, key=lambda item: item[0])
            eid = _add(evidence, source, "correlation", f"Correlation between {left} and {right}", left=left, right=right, value=value)
            findings.append(Finding("f1", f"The strongest absolute correlation in the returned matrix is {value:.2f} between {left} and {right}.", [eid]))
        else:
            limitations.append("No usable correlation values were returned.")
        return evidence, findings, "Correlation analysis was calculated from the selected numeric columns.", limitations

    if analysis == "time_series":
        rows = _rows(result)
        metric = result.get("parameters", {}).get("metric") or "row_count"
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        if len(values) >= 2:
            first, last = values[0], values[-1]
            eid = _add(evidence, source, "time_series_endpoints", f"First and last {metric} values", metric=metric, first_value=first, last_value=last)
            direction = "increased" if last > first else "decreased" if last < first else "was unchanged"
            findings.append(Finding("f1", f"{metric} {direction} from {_fmt(first)} to {_fmt(last)} across the returned time series.", [eid]))
        else:
            limitations.append("Fewer than two non-null time-series values were returned.")
        return evidence, findings, "The requested time series was calculated from the dataset.", limitations

    limitations.append(f"No deterministic answer renderer is defined for analysis type {analysis!r}.")
    return evidence, findings, "The analysis completed, but no grounded narrative finding was generated.", limitations


def _multi_step_evidence(result: dict[str, Any]) -> tuple[list[EvidenceItem], list[Finding], str, list[str]]:
    evidence: list[EvidenceItem] = []
    findings: list[Finding] = []
    limitations: list[str] = []
    steps = result.get("steps", [])
    if not isinstance(steps, list):
        raise AnswerGenerationError("Multi-step result is missing its step evidence.")

    step1 = next((s for s in steps if s.get("step_id") == "step_1"), None)
    if step1:
        rows = _rows(step1.get("result"))
        plan = step1.get("plan", {})
        metric = plan.get("metric")
        group_by = plan.get("group_by", [])
        if rows and metric and group_by and rows[0].get(metric) is not None:
            group = ", ".join(f"{c}={rows[0].get(c)}" for c in group_by)
            eid = _add(evidence, "step_1", "selected_group", "Selected highest group", group=group, metric=metric, value=rows[0][metric])
            findings.append(Finding("f1", f"The selected group is {group}, with an average {metric} of {_fmt(rows[0][metric])}.", [eid]))
        else:
            limitations.append("The selection step did not return a usable group/value.")

    trend = result.get("trend_summary")
    if isinstance(trend, dict) and trend.get("status") == "ok":
        eid = _add(
            evidence,
            result.get("final_step_id", "step_2"),
            "trend",
            "First-to-last trend",
            metric=trend.get("metric"),
            first_value=trend.get("first_value"),
            last_value=trend.get("last_value"),
            absolute_change=trend.get("absolute_change"),
            percent_change=trend.get("percent_change"),
            direction=trend.get("direction"),
        )
        pct = trend.get("percent_change")
        pct_text = "" if pct is None else f" ({pct:.2f}%)"
        findings.append(Finding("f2", f"{trend.get('metric')} {trend.get('direction')} from {_fmt(trend.get('first_value'))} to {_fmt(trend.get('last_value'))}{pct_text} across the returned time series.", [eid]))
    else:
        limitations.append("The time-series result did not contain enough usable values to establish a first-to-last trend.")

    if not findings:
        return evidence, findings, "The multi-step analysis completed, but no grounded findings could be generated.", limitations
    summary = findings[0].text
    if len(findings) > 1:
        summary += " " + findings[1].text
    return evidence, findings, summary, limitations


def validate_answer(answer: AnalystAnswer) -> AnalystAnswer:
    """Ensure every finding references evidence that actually exists."""
    known = {item.evidence_id for item in answer.evidence}
    for finding in answer.findings:
        if not finding.evidence_ids:
            raise AnswerGenerationError(f"Finding {finding.finding_id!r} has no evidence.")
        missing = set(finding.evidence_ids) - known
        if missing:
            raise AnswerGenerationError(f"Finding {finding.finding_id!r} references missing evidence: {sorted(missing)}")
    return answer


def generate_answer(result: dict[str, Any]) -> AnalystAnswer:
    """Generate a deterministic, evidence-linked answer from an agent result."""
    if not isinstance(result, dict):
        raise AnswerGenerationError("Agent result must be a dictionary.")

    if result.get("analysis") == "multi_step_reasoning":
        evidence, findings, summary, limitations = _multi_step_evidence(result)
    else:
        evidence, findings, summary, limitations = _single_step_evidence(result)

    return validate_answer(AnalystAnswer(summary, findings, evidence, limitations))
