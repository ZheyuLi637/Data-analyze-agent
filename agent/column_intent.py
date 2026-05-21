from __future__ import annotations

import re

import pandas as pd

from agent.perception import DatasetProfile


METRIC_KEYWORDS = [
    "sales",
    "revenue",
    "profit",
    "margin",
    "amount",
    "price",
    "cost",
    "unit",
    "quantity",
    "count",
    "score",
    "rating",
    "value",
    "total",
]
GROUP_KEYWORDS = [
    "region",
    "category",
    "segment",
    "product",
    "customer",
    "group",
    "type",
    "channel",
    "market",
    "department",
]
DATE_KEYWORDS = ["date", "time", "day", "month", "year", "week"]
LOW_VALUE_NUMERIC_KEYWORDS = ["id", "code", "zip", "postal"]


def select_metric(profile: DatasetProfile, goal: str = "") -> str | None:
    if not profile.numeric_columns:
        return None
    return max(
        profile.numeric_columns,
        key=lambda column: (_column_score(column, goal, METRIC_KEYWORDS) - _penalty(column), -profile.numeric_columns.index(column)),
    )


def select_group_columns(
    df: pd.DataFrame,
    profile: DatasetProfile,
    goal: str = "",
    max_columns: int = 2,
) -> list[str]:
    if not profile.categorical_columns:
        return []

    scored: list[tuple[float, str]] = []
    row_count = max(len(df), 1)
    for column in profile.categorical_columns:
        cardinality = int(df[column].nunique(dropna=True))
        if cardinality == 0:
            cardinality_score = -2.0
        elif cardinality <= min(12, max(2, row_count // 2)):
            cardinality_score = 1.5
        elif cardinality <= 30:
            cardinality_score = 0.25
        else:
            cardinality_score = -1.0
        scored.append((_column_score(column, goal, GROUP_KEYWORDS) + cardinality_score, column))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [column for _, column in scored[:max_columns]]


def select_date_column(profile: DatasetProfile, goal: str = "") -> str | None:
    if not profile.date_columns:
        return None
    return max(
        profile.date_columns,
        key=lambda column: (_column_score(column, goal, DATE_KEYWORDS), -profile.date_columns.index(column)),
    )


def describe_column_intent(df: pd.DataFrame, profile: DatasetProfile, goal: str = "") -> dict:
    metric = select_metric(profile, goal)
    groups = select_group_columns(df, profile, goal)
    date_column = select_date_column(profile, goal)
    return {
        "primary_metric": metric,
        "primary_groups": groups,
        "primary_date": date_column,
    }


def _column_score(column: str, goal: str, semantic_keywords: list[str]) -> float:
    column_tokens = _tokens(column)
    goal_tokens = _tokens(goal)
    score = 0.0

    if column_tokens & goal_tokens:
        score += 4.0
    for keyword in semantic_keywords:
        keyword_tokens = _tokens(keyword)
        if keyword_tokens & column_tokens:
            score += 2.0
        if keyword_tokens & column_tokens and keyword_tokens & goal_tokens:
            score += 3.0
    return score


def _penalty(column: str) -> float:
    tokens = _tokens(column)
    penalty = 0.0
    if tokens & set(LOW_VALUE_NUMERIC_KEYWORDS):
        penalty += 5.0
    if "discount" in tokens or "rate" in tokens:
        penalty += 1.0
    return penalty


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token}
