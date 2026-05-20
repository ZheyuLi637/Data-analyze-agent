from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/compsci767-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent.perception import DatasetProfile


@dataclass
class ToolResult:
    name: str
    title: str
    observation: str
    table: pd.DataFrame | None = None
    figure: object | None = None


def execute_tool(tool_name: str, df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    tools = {
        "dataset_summary": dataset_summary,
        "missing_value_check": missing_value_check,
        "correlation_analysis": correlation_analysis,
        "group_comparison": group_comparison,
        "trend_analysis": trend_analysis,
        "chart_generation": chart_generation,
    }
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tools[tool_name](df, profile)


def dataset_summary(df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    numeric = profile.numeric_columns
    if numeric:
        table = df[numeric].describe().transpose().round(2).reset_index()
        table = table.rename(columns={"index": "column"})
        observation = f"Generated summary statistics for {len(numeric)} numeric columns."
    else:
        table = pd.DataFrame({"column": profile.columns, "dtype": [str(df[c].dtype) for c in profile.columns]})
        observation = "No numeric columns found; returned column type summary."
    return ToolResult("dataset_summary", "Dataset Summary", observation, table=table)


def missing_value_check(df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    table = pd.DataFrame(
        {
            "column": list(profile.missing_values.keys()),
            "missing_count": list(profile.missing_values.values()),
            "missing_percent": [profile.missing_percent[c] for c in profile.missing_values],
        }
    )
    total_missing = int(table["missing_count"].sum()) if not table.empty else 0
    observation = f"Found {total_missing} missing values across the dataset."
    return ToolResult("missing_value_check", "Missing Value Check", observation, table=table)


def correlation_analysis(df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    numeric = profile.numeric_columns
    if len(numeric) < 2:
        return ToolResult(
            "correlation_analysis",
            "Correlation Analysis",
            "Skipped correlation analysis because fewer than two numeric columns were available.",
        )

    corr = df[numeric].corr(numeric_only=True).round(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    top_pairs = _top_correlations(corr)
    observation = (
        f"Computed correlations across {len(numeric)} numeric columns. "
        f"Top pairs: {'; '.join(top_pairs)}."
    )
    return ToolResult(
        "correlation_analysis",
        "Correlation Analysis",
        observation,
        table=corr.reset_index().rename(columns={"index": "column"}),
        figure=fig,
    )


def group_comparison(df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    if not profile.categorical_columns or not profile.numeric_columns:
        return ToolResult(
            "group_comparison",
            "Group Comparison",
            "Skipped group comparison because categorical and numeric columns were not both available.",
        )

    metric = profile.numeric_columns[0]
    categories = profile.categorical_columns[:2]
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
    fig, ax = plt.subplots(figsize=(6, 4))
    first_category = categories[0]
    first_group = grouped[grouped["group_column"] == first_category]
    ax.bar(first_group["group_value"].astype(str), first_group["mean"], color="#2f6f73")
    ax.set_xlabel(first_category)
    ax.set_ylabel(f"Mean {metric}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    observation = f"Compared {metric} by {', '.join(categories)}; top average groups: {'; '.join(top_summaries)}."
    return ToolResult("group_comparison", "Group Comparison", observation, table=grouped, figure=fig)


def trend_analysis(df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    if not profile.date_columns or not profile.numeric_columns:
        return ToolResult(
            "trend_analysis",
            "Trend Analysis",
            "Skipped trend analysis because date and numeric columns were not both available.",
        )

    date_column = profile.date_columns[0]
    metric = profile.numeric_columns[0]
    work = df[[date_column, metric]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column])
    trend = work.sort_values(date_column).groupby(date_column, as_index=False)[metric].sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trend[date_column], trend[metric], marker="o", color="#3659a8")
    ax.set_xlabel(date_column)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    direction = "increased" if len(trend) > 1 and trend[metric].iloc[-1] >= trend[metric].iloc[0] else "decreased"
    observation = f"Analyzed {metric} over {date_column}; the series {direction} from first to last point."
    return ToolResult("trend_analysis", "Trend Analysis", observation, table=trend, figure=fig)


def chart_generation(df: pd.DataFrame, profile: DatasetProfile) -> ToolResult:
    fig, ax = plt.subplots(figsize=(6, 4))
    if profile.numeric_columns:
        metric = profile.numeric_columns[0]
        ax.hist(df[metric].dropna(), bins=8, color="#7d5a50", edgecolor="white")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        observation = f"Generated a distribution chart for {metric}."
    elif profile.categorical_columns:
        category = profile.categorical_columns[0]
        counts = df[category].value_counts().head(10)
        ax.bar(counts.index.astype(str), counts.values, color="#7d5a50")
        ax.set_xlabel(category)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        observation = f"Generated a count chart for {category}."
    else:
        ax.text(0.5, 0.5, "No plottable columns", ha="center", va="center")
        observation = "No numeric or categorical columns were available for chart generation."
    fig.tight_layout()
    return ToolResult("chart_generation", "Chart Generation", observation, figure=fig)


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
