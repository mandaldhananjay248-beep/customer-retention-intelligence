"""Shared visual constants and lightweight CSS for the dashboard.

No business logic lives here -- only presentation: page copy, color
mappings for risk levels/priorities, and the standing disclaimer text
(kept consistent with Phase 9's recommendation documentation and Phase
11's API description).
"""

from __future__ import annotations

import streamlit as st

PAGE_TITLE = "Customer Retention Intelligence"
PAGE_SUBTITLE = "AI-powered churn prediction, explainability, and retention recommendations"

# Must match src.recommendations.next_best_action.RISK_LEVELS /
# PRIORITY_LEVELS exactly -- the dashboard displays these categories, it
# never invents, reorders, or renames them.
RISK_LEVEL_ORDER = ("low", "medium", "high", "critical")
PRIORITY_ORDER = ("low", "medium", "high", "urgent")

RISK_LEVEL_COLORS = {
    "low": "#2E7D32",
    "medium": "#F9A825",
    "high": "#EF6C00",
    "critical": "#C62828",
}

PRIORITY_COLORS = {
    "low": "#2E7D32",
    "medium": "#1565C0",
    "high": "#EF6C00",
    "urgent": "#C62828",
}

DISCLAIMER_TEXT = (
    "Model outputs are decision-support signals, not guarantees of churn. "
    "SHAP explanations describe model behavior and are not causal effects. "
    "Recommendations map observed customer signals to predefined business actions; "
    "they do not establish that an action will prevent churn."
)

CHURN_SCORE_NOTE = (
    'This model is trained with class-balancing (`class_weight="balanced"`), which shifts '
    "predicted probabilities upward relative to the true churn rate. Treat this as a **model "
    "churn score** used for ranking and prioritizing customers -- not a calibrated, "
    "real-world churn percentage."
)


def inject_base_styles() -> None:
    """A small amount of custom CSS for badges and the disclaimer footer
    -- no heavy theming, animations, or gimmicks.
    """
    st.markdown(
        """
        <style>
        .risk-badge, .priority-badge {
            display: inline-block;
            color: white;
            padding: 2px 12px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .disclaimer-box {
            font-size: 0.8rem;
            opacity: 0.75;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            padding-top: 0.6rem;
            margin-top: 1.5rem;
        }
        .section-note {
            font-size: 0.85rem;
            opacity: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(value: str, color_map: dict[str, str], css_class: str) -> str:
    color = color_map.get(value, "#616161")
    label = value.title() if isinstance(value, str) else str(value)
    return f'<span class="{css_class}" style="background-color:{color};">{label}</span>'
