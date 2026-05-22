from __future__ import annotations

import os
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/compsci767-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.column_intent import select_date_column, select_group_columns, select_metric
from agent.date_utils import parse_mixed_dates
from agent.perception import DatasetProfile, clean_numeric_series


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
        "topic_modeling": topic_modeling,
        "statistical_testing": statistical_testing,
        "predictive_modeling": predictive_modeling,
        "causal_risk_analysis": causal_risk_analysis,
        "chart_generation": chart_generation,
    }
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tools[tool_name](df, profile, goal)


def dataset_summary(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    numeric = profile.numeric_columns
    if numeric:
        table = _numeric_frame(df, numeric).describe().transpose().round(2).reset_index()
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
            _skip_message("fewer than two numeric columns were available", "Add at least two numeric fields or choose a text/group analysis."),
        )

    numeric_df = _numeric_frame(df, numeric)
    numeric_df = _drop_outlier_rows(numeric_df)
    corr = numeric_df.corr(numeric_only=True).round(2)
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
        scatter_ax.scatter(numeric_df[left], numeric_df[right], color="#3659a8", alpha=0.75)
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
        table=_correlation_pair_table(numeric_df),
        figure=fig,
    )


def group_comparison(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if not profile.categorical_columns or not profile.numeric_columns:
        return ToolResult(
            "group_comparison",
            "Group Comparison",
            _skip_message("categorical and numeric columns were not both available", "Use summary/correlation for numeric-only data or upload a grouping column."),
        )

    metric = select_metric(profile, goal) or profile.numeric_columns[0]
    categories = _valid_group_columns(df, select_group_columns(df, profile, goal, max_columns=4) or profile.categorical_columns)
    categories = categories[:2]
    if not categories:
        return ToolResult(
            "group_comparison",
            "Group Comparison",
            _skip_message("all candidate group columns have too many unique values", "Use a lower-cardinality category such as region, segment, or product type."),
        )
    work = df[categories].copy()
    work[metric] = clean_numeric_series(df[metric])
    grouped_parts = []
    top_summaries = []
    for category in categories:
        part = (
            work.groupby(category, dropna=False)[metric]
            .agg(["count", "mean", "sum", "std"])
            .round(2)
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        part = _add_mean_ci(part)
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
        part = grouped[grouped["group_column"] == category].head(10).sort_values("mean", ascending=True)
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
            _skip_message("date and numeric columns were not both available", "Run date quality check or upload a reliable date column with a numeric metric."),
        )

    date_column = select_date_column(profile, goal) or profile.date_columns[0]
    parse_success = profile.date_parse_percent.get(date_column, 0)
    if parse_success < 80:
        return ToolResult(
            "trend_analysis",
            "Trend Analysis",
            _skip_message(
                f"{date_column} has only {parse_success}% date parse success",
                "Clean date values before trusting trend analysis.",
            ),
        )
    primary_metric = select_metric(profile, goal) or profile.numeric_columns[0]
    metrics = [primary_metric] + [column for column in profile.numeric_columns if column != primary_metric][:1]
    metric = metrics[0]
    work = df[[date_column]].copy()
    for current_metric in metrics:
        work[current_metric] = clean_numeric_series(df[current_metric])
    parsed_dates = parse_mixed_dates(work[date_column]).parsed
    work = work.loc[parsed_dates.index].copy()
    work[date_column] = parsed_dates
    work = work.dropna(subset=[date_column])
    if work.empty:
        return ToolResult(
            "trend_analysis",
            "Trend Analysis",
            _skip_message(f"{date_column} could not be parsed into usable dates", "Clean date values before rerunning trend analysis."),
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
            _skip_message("no date-like columns were detected", "Add a date/time column or ask for non-time-based analysis."),
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
            _skip_message("no substantial text columns were detected", "Use a CSV column with sentence-like feedback, comments, or reviews."),
        )

    column = max(text_columns, key=lambda item: item[1])[0]
    texts = df[column].dropna().astype(str)
    table = _tfidf_keywords(texts, top_n=12)
    sentiment = _sentiment_counts(texts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    keyword_ax, sentiment_ax = axes
    keyword_ax.barh(table["keyword"][::-1], table["tfidf_score"][::-1], color="#5b7f95")
    keyword_ax.set_title(f"Top TF-IDF keywords in {column}")
    keyword_ax.set_xlabel("TF-IDF score")
    sentiment_ax.bar(sentiment.keys(), sentiment.values(), color=["#2f6f73", "#b9a44c", "#b4554b"])
    sentiment_ax.set_title("Lexicon sentiment")
    sentiment_ax.set_ylabel("Rows")
    fig.tight_layout()

    avg_words = round(texts.apply(lambda value: len(_text_tokens(value))).mean(), 1)
    top_keywords = ", ".join(table["keyword"].head(5).tolist())
    observation = (
        f"Analyzed text column {column}; average cleaned length is {avg_words} words per row. "
        f"Top TF-IDF keywords: {top_keywords}. Chart explanation: the keyword chart surfaces distinctive terms, "
        "and the sentiment chart gives a lightweight positive/neutral/negative quality signal."
    )
    return ToolResult("text_analysis", "Text Analysis", observation, table=table, figure=fig)


def topic_modeling(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    text_columns = _text_columns(df, profile)
    if not text_columns:
        return ToolResult(
            "topic_modeling",
            "Topic Modeling",
            _skip_message("no substantial text columns were detected", "Use a CSV column with sentence-like feedback, comments, or reviews."),
        )

    column = max(text_columns, key=lambda item: item[1])[0]
    texts = df[column].dropna().astype(str)
    topics = _cooccurrence_topics(texts)
    table = pd.DataFrame(topics)
    if table.empty:
        return ToolResult(
            "topic_modeling",
            "Topic Modeling",
            _skip_message(f"{column} did not contain enough repeated terms", "Collect more rows or use keyword analysis instead."),
        )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = table["topic_label"].astype(str)
    ax.barh(labels[::-1], table["support_rows"][::-1], color="#6a4fb3")
    ax.set_title(f"Topic support in {column}")
    ax.set_xlabel("Rows containing topic terms")
    fig.tight_layout()

    lead = table.iloc[0]
    observation = (
        f"Modeled recurring topics in text column {column}; strongest topic '{lead['topic_label']}' "
        f"appears in {int(lead['support_rows'])} rows. Chart explanation: the topic bar chart ranks repeated "
        "co-occurring themes so text-heavy datasets can be summarized beyond single keywords."
    )
    return ToolResult("topic_modeling", "Topic Modeling", observation, table=table, figure=fig)


def statistical_testing(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if profile.categorical_columns and profile.numeric_columns:
        metric = select_metric(profile, goal) or profile.numeric_columns[0]
        group_columns = _valid_group_columns(df, select_group_columns(df, profile, goal, max_columns=4) or profile.categorical_columns)
        if not group_columns:
            return ToolResult(
                "statistical_testing",
                "Statistical Testing",
                _skip_message("all candidate group columns have too many unique values", "Use a lower-cardinality category for group testing."),
            )
        group_column = group_columns[0]
        grouped = []
        for value, values in df.groupby(group_column, dropna=False)[metric]:
            clean_values = clean_numeric_series(values).dropna()
            if len(clean_values) < 2:
                continue
            grouped.append((value, clean_values))
        grouped.sort(key=lambda item: float(item[1].mean()), reverse=True)
        if len(grouped) >= 3:
            result = _anova_test(grouped)
            table = pd.DataFrame(
                [
                    {
                        "test": "One-way ANOVA",
                        "metric": metric,
                        "group_column": group_column,
                        "groups": len(grouped),
                        "statistic": result["statistic"],
                        "approx_p_value": result["approx_p_value"],
                        "interpretation": _significance_label(result["approx_p_value"]),
                    }
                ]
            )
            observation = (
                f"Ran ANOVA for {metric} by {group_column} across {len(grouped)} groups; F={result['statistic']:.2f} "
                f"with approximate p-value {result['approx_p_value']:.3f}. Use this as association evidence, not causality."
            )
            return ToolResult("statistical_testing", "Statistical Testing", observation, table=table)
        if len(grouped) == 2:
            left_name, left_values = grouped[0]
            right_name, right_values = grouped[-1]
            result = _welch_test(left_values, right_values)
            effect = float(left_values.mean() - right_values.mean())
            table = pd.DataFrame(
                [
                    {
                        "test": "Welch mean difference",
                        "metric": metric,
                        "group_column": group_column,
                        "comparison": f"{left_name} vs {right_name}",
                        "statistic": result["statistic"],
                        "approx_p_value": result["approx_p_value"],
                        "effect_size": round(effect, 3),
                        "interpretation": _significance_label(result["approx_p_value"]),
                    }
                ]
            )
            observation = (
                f"Ran statistical testing for {metric} by {group_column}; strongest mean gap was "
                f"{left_name} vs {right_name} with effect size {effect:.2f} and approximate p-value "
                f"{result['approx_p_value']:.3f}. Use this as evidence of group difference, not causality."
            )
            return ToolResult("statistical_testing", "Statistical Testing", observation, table=table)

    if len(profile.numeric_columns) >= 2:
        left, right = _strongest_numeric_pair(df, profile)
        if left and right:
            work = _numeric_frame(df, [left, right]).dropna()
            if len(work) >= 3:
                r = float(work[left].corr(work[right]))
                p_value = _correlation_p_value(r, len(work))
                table = pd.DataFrame(
                    [
                        {
                            "test": "Pearson correlation",
                            "left": left,
                            "right": right,
                            "n": len(work),
                            "statistic": round(r, 3),
                            "approx_p_value": p_value,
                            "bonferroni_p_value": p_value,
                            "significant_after_bonferroni": p_value < 0.05,
                            "interpretation": _significance_label(p_value),
                        }
                    ]
                )
                observation = (
                    f"Ran statistical testing for the strongest numeric pair {left} vs {right}; Pearson r={r:.2f} "
                    f"with approximate p-value {p_value:.3f}. This tests association strength, not a causal effect."
                )
                return ToolResult("statistical_testing", "Statistical Testing", observation, table=table)

    return ToolResult(
        "statistical_testing",
        "Statistical Testing",
        _skip_message("the dataset did not contain enough comparable numeric observations", "Add more numeric rows or a lower-cardinality group column."),
    )


def predictive_modeling(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if not profile.numeric_columns:
        return ToolResult(
            "predictive_modeling",
            "Predictive Modeling",
            _skip_message("no numeric target column was detected", "Add a numeric target such as sales, profit, score, or amount."),
        )

    target = select_metric(profile, goal) or profile.numeric_columns[0]
    feature_columns = _candidate_prediction_features(df, profile, target)
    if not feature_columns and profile.date_columns:
        date_column = select_date_column(profile, goal) or profile.date_columns[0]
        parsed = parse_mixed_dates(df[date_column]).parsed
        time_index = parsed.apply(lambda value: value.toordinal() if pd.notna(value) else np.nan)
        work = pd.DataFrame({"_time_index": time_index, target: clean_numeric_series(df[target])}).dropna()
        feature_columns = ["_time_index"]
    else:
        work = _numeric_frame(df, [target] + feature_columns).dropna()

    if len(feature_columns) == 0 or len(work) < 3:
        return ToolResult(
            "predictive_modeling",
            "Predictive Modeling",
            _skip_message(f"{target} did not have enough usable feature rows", "Add at least one usable numeric feature with three or more complete rows."),
        )

    feature, feature_reason = _best_prediction_feature(work, target, feature_columns)
    model = _fit_simple_linear_model(work[feature], work[target])
    validation = _train_test_metrics(work, target, feature)
    table = pd.DataFrame(
        [
            {
                "target": target,
                "feature": feature,
                "feature_reason": feature_reason,
                "rows_used": len(work),
                "slope": model["slope"],
                "intercept": model["intercept"],
                "r_squared": model["r_squared"],
                "mae": model["mae"],
                "train_rows": validation["train_rows"],
                "test_rows": validation["test_rows"],
                "test_mae": validation["test_mae"],
                "test_r_squared": validation["test_r_squared"],
                "validation_note": validation["validation_note"],
                "next_feature_value": model["next_feature_value"],
                "next_prediction": model["next_prediction"],
            }
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(work[feature], work[target], color="#3659a8", alpha=0.7, label="observed")
    x_values = np.linspace(float(work[feature].min()), float(work[feature].max()), 50)
    y_values = model["intercept"] + model["slope"] * x_values
    ax.plot(x_values, y_values, color="#b4554b", label="linear fit")
    ax.set_title(f"Predict {target} from {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.legend()
    fig.tight_layout()

    observation = (
        f"Built a simple predictive model for {target} using {feature}; R-squared={model['r_squared']:.2f}, "
        f"MAE={model['mae']:.2f}, test MAE={validation['test_mae']}, and next-step prediction is {model['next_prediction']:.2f}. "
        "This is a baseline forecast for planning, not a production model."
    )
    return ToolResult("predictive_modeling", "Predictive Modeling", observation, table=table, figure=fig)


def causal_risk_analysis(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if not profile.numeric_columns:
        return ToolResult(
            "causal_risk_analysis",
            "Causal Risk Analysis",
            _skip_message("no numeric outcome column was detected", "Add a numeric outcome before auditing causal risk."),
        )

    outcome = select_metric(profile, goal) or profile.numeric_columns[0]
    rows = []
    if profile.categorical_columns:
        group_column = (select_group_columns(df, profile, goal, max_columns=1) or profile.categorical_columns[:1])[0]
        grouped = df.groupby(group_column, dropna=False)[outcome].agg(["count", "mean"]).dropna().sort_values("mean", ascending=False)
        if len(grouped) >= 2:
            high = grouped.iloc[0]
            low = grouped.iloc[-1]
            rows.append(
                {
                    "claim_type": "group association",
                    "outcome": outcome,
                    "driver": group_column,
                    "observed_association": f"{grouped.index[0]} mean {high['mean']:.2f} vs {grouped.index[-1]} mean {low['mean']:.2f}",
                    "causal_guardrail": "Observational difference only; randomized assignment or controls are required for causality.",
                }
            )
    for feature in [column for column in profile.numeric_columns if column != outcome][:3]:
        work = _numeric_frame(df, [feature, outcome]).dropna()
        if len(work) < 3:
            continue
        rows.append(
            {
                "claim_type": "numeric association",
                "outcome": outcome,
                "driver": feature,
                "observed_association": f"correlation {work[feature].corr(work[outcome]):.2f}",
                "causal_guardrail": "Association may be confounded by seasonality, segment mix, pricing, or omitted variables.",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return ToolResult(
            "causal_risk_analysis",
            "Causal Risk Analysis",
            _skip_message("no comparable driver/outcome relationship was available", "Add a driver column and numeric outcome with enough complete rows."),
        )

    possible_confounders = [column for column in profile.columns if column != outcome][:5]
    observation = (
        f"Audited causal risk for outcome {outcome}; found {len(table)} observational association(s). "
        f"Possible confounders to control before causal claims: {', '.join(possible_confounders)}. "
        "Conclusion: the agent can flag causal hypotheses, but it does not claim causality from this CSV alone."
    )
    return ToolResult("causal_risk_analysis", "Causal Risk Analysis", observation, table=table)


def chart_generation(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> ToolResult:
    if profile.numeric_columns:
        metric = select_metric(profile, goal) or profile.numeric_columns[0]
        values = df[metric].dropna()
        values = clean_numeric_series(df[metric]).dropna()
        if values.empty:
            return ToolResult(
                "chart_generation",
                "Chart Generation",
                _skip_message(f"{metric} has no non-missing values", "Choose another metric or clean missing numeric values."),
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
MAX_GROUP_CARDINALITY = 30


def _skip_message(reason: str, next_step: str) -> str:
    return f"Skipped because {reason}. Next step: {next_step}"


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: clean_numeric_series(df[column]) for column in columns if column in df.columns})


def _drop_outlier_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column in frame.columns:
        values = frame[column].dropna()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= frame[column].isna() | frame[column].between(lower, upper)
    filtered = frame.loc[mask]
    return filtered if len(filtered.dropna(how="all")) >= 3 else frame


def _valid_group_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    row_count = max(len(df), 1)
    threshold = min(MAX_GROUP_CARDINALITY, max(2, int(row_count * 0.5)))
    valid = []
    for column in columns:
        cardinality = int(df[column].nunique(dropna=True))
        if 1 < cardinality <= threshold:
            valid.append(column)
    return valid


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
    return [
        token
        for token in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if token not in STOPWORDS
    ]


def _tfidf_keywords(texts: pd.Series, top_n: int = 12) -> pd.DataFrame:
    documents = [_text_tokens(text) for text in texts]
    document_count = max(len(documents), 1)
    term_frequency = Counter(token for document in documents for token in document)
    document_frequency = Counter(token for document in documents for token in set(document))
    rows = []
    for token, count in term_frequency.items():
        df_count = document_frequency[token]
        idf = math.log((1 + document_count) / (1 + df_count)) + 1
        rows.append(
            {
                "keyword": token,
                "tfidf_score": round(float(count * idf), 3),
                "document_count": int(df_count),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["keyword", "tfidf_score", "document_count"])
    return table.sort_values(["tfidf_score", "document_count"], ascending=False).head(top_n).reset_index(drop=True)


def _cooccurrence_topics(texts: pd.Series, max_topics: int = 5) -> list[dict]:
    row_tokens = [set(_text_tokens(text)) for text in texts if _text_tokens(text)]
    token_counts = Counter(token for tokens in row_tokens for token in tokens)
    vocabulary = [token for token, count in token_counts.most_common(40) if count >= 2]
    if not vocabulary:
        vocabulary = [token for token, _ in token_counts.most_common(12)]

    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for tokens in row_tokens:
        present = sorted(token for token in tokens if token in vocabulary)
        for index, left in enumerate(present):
            for right in present[index + 1 :]:
                pair_counts[(left, right)] += 1

    topics: list[dict] = []
    used_terms: set[str] = set()
    for (left, right), count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
        if left in used_terms and right in used_terms:
            continue
        support = sum(1 for tokens in row_tokens if left in tokens or right in tokens)
        example = _topic_example(texts, {left, right})
        topics.append(
            {
                "topic_label": f"{left} / {right}",
                "terms": f"{left}, {right}",
                "cooccurrence_count": count,
                "support_rows": support,
                "example_text": example,
            }
        )
        used_terms.update({left, right})
        if len(topics) >= max_topics:
            break

    if topics:
        return topics

    for token, count in token_counts.most_common(max_topics):
        topics.append(
            {
                "topic_label": token,
                "terms": token,
                "cooccurrence_count": 0,
                "support_rows": count,
                "example_text": _topic_example(texts, {token}),
            }
        )
    return topics


def _topic_example(texts: pd.Series, terms: set[str]) -> str:
    for text in texts:
        tokens = set(_text_tokens(text))
        if terms & tokens:
            return str(text)[:140]
    return ""


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


def _correlation_pair_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = list(frame.columns)
    pair_count = max(1, len(columns) * (len(columns) - 1) // 2)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            work = frame[[left, right]].dropna()
            if len(work) < 3:
                continue
            r = float(work[left].corr(work[right]))
            p_value = _correlation_p_value(r, len(work))
            adjusted = round(min(1.0, p_value * pair_count), 4)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n": len(work),
                    "pearson_r": round(r, 3),
                    "approx_p_value": p_value,
                    "bonferroni_p_value": adjusted,
                    "significant_after_bonferroni": adjusted < 0.05,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["left", "right", "n", "pearson_r", "approx_p_value", "bonferroni_p_value", "significant_after_bonferroni"])
    return table.sort_values("pearson_r", key=lambda values: values.abs(), ascending=False).reset_index(drop=True)


def _parse_pair(pair_text: str) -> tuple[str, str] | None:
    if " vs " not in pair_text:
        return None
    left, rest = pair_text.split(" vs ", 1)
    right = rest.split(" (", 1)[0]
    if not left or not right:
        return None
    return left, right


def _strongest_numeric_pair(df: pd.DataFrame, profile: DatasetProfile) -> tuple[str | None, str | None]:
    corr = _numeric_frame(df, profile.numeric_columns).corr(numeric_only=True)
    best: tuple[float, str | None, str | None] = (0.0, None, None)
    for index, left in enumerate(profile.numeric_columns):
        for right in profile.numeric_columns[index + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            if abs(float(value)) > best[0]:
                best = (abs(float(value)), left, right)
    return best[1], best[2]


def _welch_test(left: pd.Series, right: pd.Series) -> dict[str, float]:
    left_mean = float(left.mean())
    right_mean = float(right.mean())
    left_var = float(left.var(ddof=1))
    right_var = float(right.var(ddof=1))
    left_n = len(left)
    right_n = len(right)
    standard_error = math.sqrt((left_var / left_n) + (right_var / right_n))
    statistic = 0.0 if standard_error == 0 else (left_mean - right_mean) / standard_error
    return {
        "statistic": round(statistic, 3),
        "approx_p_value": _normal_two_sided_p(statistic),
    }


def _anova_test(grouped: list[tuple[object, pd.Series]]) -> dict[str, float]:
    values = [series.astype(float) for _, series in grouped if len(series) >= 2]
    all_values = pd.concat(values, ignore_index=True)
    grand_mean = float(all_values.mean())
    between = sum(len(series) * (float(series.mean()) - grand_mean) ** 2 for series in values)
    within = sum(float(((series - float(series.mean())) ** 2).sum()) for series in values)
    df_between = max(len(values) - 1, 1)
    df_within = max(len(all_values) - len(values), 1)
    ms_between = between / df_between
    ms_within = within / df_within if within > 0 else 1e-9
    statistic = ms_between / ms_within
    approx_p = round(max(0.0, min(1.0, math.exp(-0.5 * statistic * df_between))), 4)
    return {"statistic": round(float(statistic), 3), "approx_p_value": approx_p}


def _correlation_p_value(r: float, n: int) -> float:
    if n <= 2 or abs(r) >= 1:
        return 0.0 if abs(r) >= 1 else 1.0
    statistic = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
    return _normal_two_sided_p(statistic)


def _normal_two_sided_p(statistic: float) -> float:
    tail = 1 - _normal_cdf(abs(statistic))
    return round(max(0.0, min(1.0, 2 * tail)), 4)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _significance_label(p_value: float) -> str:
    if p_value < 0.05:
        return "statistically notable at 0.05 using normal approximation"
    if p_value < 0.1:
        return "directional signal, not conventionally significant"
    return "not statistically notable"


def _candidate_prediction_features(df: pd.DataFrame, profile: DatasetProfile, target: str) -> list[str]:
    candidates = []
    for column in profile.numeric_columns:
        if column == target:
            continue
        missing_percent = profile.missing_percent.get(column, 0)
        unique_count = int(df[column].nunique(dropna=True))
        name = column.lower()
        if missing_percent > 40:
            continue
        if unique_count >= max(10, int(len(df) * 0.9)) and any(token in name for token in ("id", "code", "zip", "postal")):
            continue
        candidates.append(column)
    return candidates


def _best_prediction_feature(work: pd.DataFrame, target: str, feature_columns: list[str]) -> tuple[str, str]:
    best_feature = feature_columns[0]
    best_score = -1.0
    for feature in feature_columns:
        corr = work[feature].corr(work[target])
        score = 0.0 if pd.isna(corr) else abs(float(corr))
        if score > best_score:
            best_feature = feature
            best_score = score
    return best_feature, f"selected highest absolute correlation among usable low-missing numeric features ({best_score:.2f})"


def _fit_simple_linear_model(feature: pd.Series, target: pd.Series) -> dict[str, float]:
    x = feature.astype(float).to_numpy()
    y = target.astype(float).to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    predictions = intercept + slope * x
    residuals = y - predictions
    total_variance = float(np.sum((y - y.mean()) ** 2))
    residual_variance = float(np.sum(residuals**2))
    r_squared = 0.0 if total_variance == 0 else 1 - residual_variance / total_variance
    mae = float(np.mean(np.abs(residuals)))
    step = float(np.median(np.diff(np.sort(x)))) if len(x) > 1 else 1.0
    if step == 0:
        step = 1.0
    next_feature_value = float(x.max() + step)
    next_prediction = float(intercept + slope * next_feature_value)
    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_squared), 4),
        "mae": round(mae, 4),
        "next_feature_value": round(next_feature_value, 4),
        "next_prediction": round(next_prediction, 4),
    }


def _train_test_metrics(work: pd.DataFrame, target: str, feature: str) -> dict:
    if len(work) < 8:
        return {
            "train_rows": len(work),
            "test_rows": 0,
            "test_mae": "n/a",
            "test_r_squared": "n/a",
            "validation_note": "insufficient rows for train/test; fitted baseline only",
        }
    ordered = work[[feature, target]].dropna().reset_index(drop=True)
    split_index = max(1, int(len(ordered) * 0.8))
    train = ordered.iloc[:split_index]
    test = ordered.iloc[split_index:]
    model = _fit_simple_linear_model(train[feature], train[target])
    predictions = model["intercept"] + model["slope"] * test[feature].astype(float).to_numpy()
    actual = test[target].astype(float).to_numpy()
    residuals = actual - predictions
    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    residual_variance = float(np.sum(residuals**2))
    r_squared = 0.0 if total_variance == 0 else 1 - residual_variance / total_variance
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_mae": round(float(np.mean(np.abs(residuals))), 4),
        "test_r_squared": round(float(r_squared), 4),
        "validation_note": "80/20 train/test split in original row order",
    }


def _add_mean_ci(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    std_error = table.apply(
        lambda row: 0.0 if row["count"] <= 1 or pd.isna(row.get("std")) else float(row["std"]) / math.sqrt(float(row["count"])),
        axis=1,
    )
    table["ci_low"] = (table["mean"] - 1.96 * std_error).round(2)
    table["ci_high"] = (table["mean"] + 1.96 * std_error).round(2)
    return table
