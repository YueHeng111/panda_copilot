from __future__ import annotations

from typing import Dict, List

import pandas as pd


KEY_COMPARE_COLS = [
    "company_id",
    "company_name",
    "industry",
    "region",
    "relationship_stage",
    "transition_project",
    "panda_score",
    "green_loan_fit_score",
    "financial_resilience_score",
    "green_impact_score",
    "governance_score",
    "documentation_score",
    "pd_estimate",
    "recommended_product",
    "green_loan_fit",
]


def portfolio_snapshot(scored: pd.DataFrame) -> Dict[str, object]:
    return {
        "n_companies": int(len(scored)),
        "avg_panda_score": float(scored["panda_score"].mean()),
        "avg_green_loan_fit_score": float(scored["green_loan_fit_score"].mean()),
        "fit_A_count": int((scored["green_loan_fit"].str.startswith("A")).sum()),
        "high_pd_count": int((scored["pd_estimate"] >= 0.09).sum()),
        "top_products": scored["recommended_product"].value_counts().to_dict(),
    }


def company_comparison(scored: pd.DataFrame, company_ids: List[str]) -> pd.DataFrame:
    return scored[scored["company_id"].isin(company_ids)][KEY_COMPARE_COLS].copy()


def industry_comparison(scored: pd.DataFrame, benchmarks: pd.DataFrame, company_id: str) -> Dict[str, object]:
    row = scored[scored["company_id"] == company_id].iloc[0]
    bench = benchmarks[benchmarks["industry"] == row["industry"]]
    means = bench.set_index("metric")["mean"].to_dict()
    return {"company": row.to_dict(), "industry_mean": means}


def top_n_by_metric(scored: pd.DataFrame, metric: str, ascending: bool, n: int = 10) -> pd.DataFrame:
    return scored.sort_values(metric, ascending=ascending).head(n).copy()


def missing_documents_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[["company_id", "company_name", "recommended_product", "missing_documents"]].copy()


REPORT_CATALOG = [
    "企業綠色授信申請摘要",
    "銀行前審 Memo",
    "缺件與補件清單",
    "綠色貸款適配建議",
    "專案 CAPEX / 節能效益摘要",
    "授信條件與 KPI 建議草案",
    "貸後追蹤月報",
    "供應鏈議合建議書",
    "訪廠訪談提綱",
    "交叉銷售與合作夥伴推薦",
]
