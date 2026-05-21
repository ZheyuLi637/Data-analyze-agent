from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/compsci767-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent.column_intent import select_date_column, select_group_columns, select_metric
from agent.date_utils import parse_mixed_dates
from agent.perception import DatasetProfile


@dataclass
class ToolResult:
    name: str
    title: str
    observation: str
    table: pd.DataFrame | None = None
    figure: object | None = None


def execute_tool(tool_name: str, df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    tools = {
        "dataset_summary": dataset_summary,
        "missing_value_check": missing_value_check,
        "correlation_analysis": correlation_analysis,
        "group_comparison": group_comparison,
        "trend_analysis": trend_analysis,
        "date_quality_check": date_quality_check,
        "text_analysis": text_analysis,
        "chart_generation": chart_generation,
    }
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tools[tool_name](df, profile, goal)


def dataset_summary(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    numeric = profile.numeric_columns
    if numeric:
        table = df[numeric].describe().transpose().round(2).reset_index()
        table = table.rename(columns={"index": "column"})
        observation = f"Generated summary statistics for {len(numeric)} numeric columns."
    else:
        table = pd.DataFrame({"column": profile.columns, "dtype": [str(df[c].dtype) for c in profile.columns]})
        observation = "No numeric columns found; returned column type summary."
    return ToolResult("dataset_summary", "Dataset Summary", observation, table=table)


def missing_value_check(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    table = pd.DataFrame(
        {
            "column": list(profile.missing_values.keys()),
            "missing_count": list(profile.missing_values.values()),
            "missing_percent": [profile.missing_percent[c] for c in profile.missing_values],
        }
    )
    table["severity"] = table["missing_percent"].apply(_missing_severity)
    table = table.sort_values(["missing_percent", "missing_count"], ascending=False).reset_index(drop=True)
    total_missing = int(table["missing_count"].sum()) if not table.empty else 0
    rows_with_missing = int(df.isna().any(axis=1).sum())
    row_percent = round((rows_with_missing / max(len(df), 1)) * 100, 1)
    affected_columns = table[table["missing_count"] > 0]
    if affected_columns.empty:
        observation = "Found 0 missing values across the dataset."
    else:
        top = affected_columns.iloc[0]
        high_risk_columns = affected_columns[affected_columns["missing_percent"] >= 30]["column"].astype(str).tolist()
        group_note = _missing_group_note(df, profile, goal)
        risk_note = (
            f" High-risk columns: {', '.join(high_risk_columns)}."
            if high_risk_columns
            else " No column exceeds the 30% high-risk threshold."
        )
        observation = (
            f"Found {total_missing} missing values across the dataset; {rows_with_missing} rows "
            f"({row_percent}%) contain at least one missing value. Highest missing column: "
            f"{top['column']} ({top['missing_count']} values, {top['missing_percent']}%)."
            f"{risk_note}{group_note}"
        )
    return ToolResult("missing_value_check", "Missing Value Check", observation, table=table)


def correlation_analysis(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    numeric = profile.numeric_columns
    if len(numeric) < 2:
        return ToolResult(
            "correlation_analysis",
            "Correlation Analysis",
            "Skipped correlation analysis because fewer than two numeric columns were available.",
        )

    corr = df[numeric].corr(numeric_only=True).round(2)
    top_pairs = _top_correlations(corr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    heatmap_ax, scatter_ax = axes
    image = heatmap_ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    heatmap_ax.set_title("Correlation heatmap")
    heatmap_ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    heatmap_ax.set_yticks(range(len(corr.index)), corr.index)
    for row_index, row_name in enumerate(corr.index):
        for column_index, column_name in enumerate(corr.columns):
            heatmap_ax.text(column_index, row_index, corr.loc[row_name, column_name], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04)

    pair = _parse_pair(top_pairs[0])
    if pair:
        left, right = pair
        scatter_ax.scatter(df[left], df[right], color="#3659a8", alpha=0.75)
        scatter_ax.set_title(f"Strongest pair: {left} vs {right}")
        scatter_ax.set_xlabel(left)
        scatter_ax.set_ylabel(right)
    else:
        scatter_ax.text(0.5, 0.5, "No pair available", ha="center", va="center")
        scatter_ax.set_axis_off()
    fig.tight_layout()

    observation = (
        f"Computed correlations across {len(numeric)} numeric columns. "
        f"Top pairs: {'; '.join(top_pairs)}. Chart explanation: the heatmap highlights relationship strength "
        "across all numeric columns, and the scatter plot checks the strongest pair for visual consistency."
    )
    return ToolResult(
        "correlation_analysis",
        "Correlation Analysis",
        observation,
        table=corr.reset_index().rename(columns={"index": "column"}),
        figure=fig,
    )


def group_comparison(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if not profile.categorical_columns or not profile.numeric_columns:
        return ToolResult(
            "group_comparison",
            "Group Comparison",
            "Skipped group comparison because categorical and numeric columns were not both available.",
        )

    metric = select_metric(profile, goal) or profile.numeric_columns[0]
    categories = select_group_columns(df, profile, goal, max_columns=2) or profile.categorical_columns[:2]
    grouped_parts = []
    top_summaries = []
    for category in categories:
        part = (
            df.groupby(category, dropna=False)[metric]
            .agg(["count", "mean", "sum"])
            .round(2)
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        part.insert(0, "group_column", category)
        part = part.rename(columns={category: "group_value"})
        grouped_parts.append(part)
        if not part.empty:
            top_summaries.append(f"{category}: {part.iloc[0]['group_value']}")

    grouped = pd.concat(grouped_parts, ignore_index=True)
    fig, axes = plt.subplots(1, len(categories), figsize=(5.5 * len(categories), 4.2))
    if len(categories) == 1:
        axes = [axes]
    for ax, category in zip(axes, categories):
        part = grouped[grouped["group_column"] == category].sort_values("mean", ascending=True)
        ax.barh(part["group_value"].astype(str), part["mean"], color="#2f6f73")
        ax.set_title(f"Mean {metric} by {category}")
        ax.set_xlabel(f"Mean {metric}")
    fig.tight_layout()

    observation = (
        f"Compared {metric} by {', '.join(categories)}; top average groups: {'; '.join(top_summaries)}. "
        "Chart explanation: side-by-side bar charts use longer bars to indicate groups with higher average metric values, making the strongest "
        "groups visible without reading every table row."
    )
    return ToolResult("group_comparison", "Group Comparison", observation, table=grouped, figure=fig)


def trend_analysis(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if not profile.date_columns or not profile.numeric_columns:
        return ToolResult(
            "trend_analysis",
            "Trend Analysis",
            "Skipped trend analysis because date and numeric columns were not both available.",
        )

    date_column = select_date_column(profile, goal) or profile.date_columns[0]
    primary_metric = select_metric(profile, goal) or profile.numeric_columns[0]
    metrics = [primary_metric] + [column for column in profile.numeric_columns if column != primary_metric][:1]
    metric = metrics[0]
    work = df[[date_column] + metrics].copy()
    parsed_dates = parse_mixed_dates(work[date_column]).parsed
    work = work.loc[parsed_dates.index].copy()
    work[date_column] = parsed_dates
    work = work.dropna(subset=[date_column])
    if work.empty:
        return ToolResult(
            "trend_analysis",
            "Trend Analysis",
            f"Skipped trend analysis because {date_column} could not be parsed into usable dates.",
        )
    trend = work.sort_values(date_column).groupby(date_column, as_index=False)[metrics].sum()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = ["#3659a8", "#b4554b"]
    for index, current_metric in enumerate(metrics):
        ax.plot(trend[date_column], trend[current_metric], marker="o", color=colors[index % len(colors)], label=current_metric)
    if len(trend) >= 3:
        rolling = trend[metric].rolling(window=3, min_periods=1).mean()
        ax.plot(trend[date_column], rolling, linestyle="--", color="#222222", label=f"{metric} rolling avg")
    ax.legend()
    ax.set_title("Trend with comparison metric")
    ax.set_xlabel(date_column)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    direction = "increased" if len(trend) > 1 and trend[metric].iloc[-1] >= trend[metric].iloc[0] else "decreased"
    if len(trend) > 1:
        value_note = f" from {trend[metric].iloc[0]:.2f} to {trend[metric].iloc[-1]:.2f}"
    else:
        value_note = ""
    observation = (
        f"Analyzed {', '.join(metrics)} over {date_column}; {metric} {direction}{value_note} from first to last point. "
        "Chart explanation: the multi-metric trend line chart shows movement over time, the comparison line adds context, and the "
        "rolling average smooths short-term noise when enough points are available."
    )
    return ToolResult("trend_analysis", "Trend Analysis", observation, table=trend, figure=fig)


def date_quality_check(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    candidates = profile.date_parse_percent or {
        column: parse_mixed_dates(df[column]).success_percent
        for column in profile.columns
        if any(token in column.lower() for token in ("date", "time", "day"))
    }
    if not candidates:
        return ToolResult(
            "date_quality_check",
            "Date Quality Check",
            "Skipped date quality check because no date-like columns were detected.",
        )

    rows = []
    for column, success_percent in candidates.items():
        parsed = parse_mixed_dates(df[column])
        rows.append(
            {
                "column": column,
                "parse_success_percent": success_percent,
                "invalid_examples": ", ".join(parsed.invalid_examples) if parsed.invalid_examples else "",
            }
        )
    table = pd.DataFrame(rows).sort_values("parse_success_percent", ascending=False)
    best = table.iloc[0]
    observation = (
        f"Checked date quality for {len(table)} date-like column(s). Best candidate: {best['column']} "
        f"with {best['parse_success_percent']}% parse success."
    )
    if best["parse_success_percent"] < 80:
        observation += " Date cleanup is recommended before trusting trend conclusions."
    else:
        observation += " The parsed dates are usable for a cautious trend analysis."
    return ToolResult("date_quality_check", "Date Quality Check", observation, table=table)


def text_analysis(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    text_columns = _text_columns(df, profile)
    if not text_columns:
        return ToolResult(
            "text_analysis",
            "Text Analysis",
            "Skipped text analysis because no substantial text columns were detected.",
        )

    column = max(text_columns, key=lambda item: item[1])[0]
    texts = df[column].dropna().astype(str)
    tokens = _text_tokens(" ".join(texts))
    keyword_counts = pd.Series(tokens).value_counts().head(12)
    table = keyword_counts.reset_index()
    table.columns = ["keyword", "count"]
    sentiment = _sentiment_counts(texts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    keyword_ax, sentiment_ax = axes
    keyword_ax.barh(table["keyword"][::-1], table["count"][::-1], color="#5b7f95")
    keyword_ax.set_title(f"Top keywords in {column}")
    keyword_ax.set_xlabel("Count")
    sentiment_ax.bar(sentiment.keys(), sentiment.values(), color=["#2f6f73", "#b9a44c", "#b4554b"])
    sentiment_ax.set_title("Lexicon sentiment")
    sentiment_ax.set_ylabel("Rows")
    fig.tight_layout()

    avg_words = round(texts.apply(lambda value: len(_text_tokens(value))).mean(), 1)
    top_keywords = ", ".join(table["keyword"].head(5).tolist())
    observation = (
        f"Analyzed text column {column}; average cleaned length is {avg_words} words per row. "
        f"Top keywords: {top_keywords}. Chart explanation: the keyword chart surfaces repeated themes, "
        "and the sentiment chart gives a lightweight positive/neutral/negative quality signal."
    )
    return ToolResult("text_analysis", "Text Analysis", observation, table=table, figure=fig)


def chart_generation(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if profile.numeric_columns:
        metric = select_metric(profile, goal) or profile.numeric_columns[0]
        values = df[metric].dropna()
        if values.empty:
            return ToolResult(
                "chart_generation",
                "Chart Generation",
                f"Skipped chart generation because {metric} has no non-missing values.",
            )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        hist_ax, box_ax = axes
        hist_ax.hist(values, bins=8, color="#7d5a50", edgecolor="white")
        hist_ax.set_title(f"{metric} distribution")
        hist_ax.set_xlabel(metric)
        hist_ax.set_ylabel("Count")
        box_ax.boxplot(values, orientation="vertical", patch_artist=True, boxprops={"facecolor": "#d8b4a6"})
        box_ax.set_title(f"{metric} spread")
        box_ax.set_ylabel(metric)
        observation = (
            f"Generated distribution and boxplot views for {metric}. Chart explanation: the histogram shows common "
            f"value ranges, while the boxplot shows median, spread, and possible outliers from {values.min():.2f} to {values.max():.2f}."
        )
    elif profile.categorical_columns:
        category = select_group_columns(df, profile, goal, max_columns=1)[0]
        counts = df[category].value_counts().head(10)
        shares = (counts / counts.sum() * 100).round(1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        count_ax, share_ax = axes
        count_ax.bar(counts.index.astype(str), counts.values, color="#7d5a50")
        count_ax.set_title(f"{category} counts")
        count_ax.set_xlabel(category)
        count_ax.set_ylabel("Count")
        count_ax.tick_params(axis="x", rotation=30)
        share_ax.barh(shares.index.astype(str), shares.values, color="#5b7f95")
        share_ax.set_title(f"{category} share")
        share_ax.set_xlabel("Percent")
        top_label = str(counts.index[0]) if not counts.empty else "n/a"
        observation = (
            f"Generated count and share charts for {category}. Chart explanation: the tallest bar is {top_label}, "
            "and the share chart shows whether one category dominates the dataset."
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No plottable columns", ha="center", va="center")
        observation = "No numeric or categorical columns were available for chart generation."
    fig.tight_layout()
    return ToolResult("chart_generation", "Chart Generation", observation, figure=fig)


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "was",
    "were",
    "but",
    "this",
    "that",
    "are",
    "our",
    "too",
    "very",
    "from",
    "have",
    "has",
    "into",
    "data",
}
POSITIVE_WORDS = {"smooth", "easy", "useful", "quickly", "solved", "clear", "good", "fast", "great", "positive"}
NEGATIVE_WORDS = {"slow", "confusing", "issue", "problem", "bad", "missing", "negative", "hard", "error", "delay"}


def _text_columns(df: pd.DataFrame, profile: DatasetProfile) -> list[tuple[str, float]]:
    columns: list[tuple[str, float]] = []
    for column in profile.categorical_columns:
        texts = df[column].dropna().astype(str)
        if texts.empty:
            continue
        avg_length = float(texts.str.len().mean())
        avg_words = float(texts.apply(lambda value: len(_text_tokens(value))).mean())
        if avg_length >= 20 or avg_words >= 4:
            columns.append((column, avg_words))
    return columns


def _text_tokens(text: str) -> list[str]:
    import re

    return [
        token
        for token in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if token not in STOPWORDS
    ]


def _sentiment_counts(texts: pd.Series) -> dict[str, int]:
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for text in texts:
        tokens = set(_text_tokens(text))
        score = len(tokens & POSITIVE_WORDS) - len(tokens & NEGATIVE_WORDS)
        if score > 0:
            counts["positive"] += 1
        elif score < 0:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1
    return counts


def _missing_severity(percent: float) -> str:
    if percent >= 30:
        return "high"
    if percent >= 10:
        return "medium"
    if percent > 0:
        return "low"
    return "none"


def _missing_group_note(df: pd.DataFrame, profile: DatasetProfile, goal: str) -> str:
    groups = select_group_columns(df, profile, goal, max_columns=1)
    if not groups:
        return ""
    group = groups[0]
    work = pd.DataFrame(
        {
            "group_value": df[group].fillna("(missing group)").astype(str),
            "row_has_missing": df.isna().any(axis=1),
        }
    )
    grouped = (
        work.groupby("group_value")["row_has_missing"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"sum": "rows_with_missing"})
    )
    grouped["missing_row_percent"] = (grouped["rows_with_missing"] / grouped["count"] * 100).round(1)
    grouped = grouped.sort_values(["missing_row_percent", "rows_with_missing"], ascending=False)
    if grouped.empty:
        return ""
    top = grouped.iloc[0]
    return (
        f" Missing rows are most concentrated in {group}={top['group_value']} "
        f"({int(top['rows_with_missing'])}/{int(top['count'])} rows, {top['missing_row_percent']}%)."
    )


def _top_correlations(corr: pd.DataFrame) -> list[str]:
    pairs: list[tuple[float, str]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append((abs(float(value)), f"{left} vs {right} ({float(value):.2f})"))
    pairs.sort(reverse=True, key=lambda item: item[0])
    return [pair for _, pair in pairs[:3]] or ["n/a"]


def _parse_pair(pair_text: str) -> tuple[str, str] | None:
    if " vs " not in pair_text:
        return None
    left, rest = pair_text.split(" vs ", 1)
    right = rest.split(" (", 1)[0]
    if not left or not right:
        return None
    return left, right
