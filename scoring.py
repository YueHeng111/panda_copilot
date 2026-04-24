from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from config import CONFIG
from data_simulator import DOC_FIELDS, save_company_documents


def load_datasets() -> Dict[str, pd.DataFrame]:
    return {
        "master": pd.read_csv(CONFIG.data_dir / "company_master.csv"),
        "financials_annual": pd.read_csv(CONFIG.data_dir / "financials_annual.csv"),
        "monthly_panel": pd.read_csv(CONFIG.data_dir / "monthly_panel.csv", parse_dates=["date"]),
        "esg_documents": pd.read_csv(CONFIG.data_dir / "esg_documents.csv"),
        "banking_metrics": pd.read_csv(CONFIG.data_dir / "banking_metrics.csv"),
    }


def _normalize_pos(series: pd.Series) -> pd.Series:
    lo, hi = series.quantile(0.05), series.quantile(0.95)
    scaled = (series.clip(lo, hi) - lo) / max(hi - lo, 1e-9)
    return scaled.clip(0, 1)


def _normalize_neg(series: pd.Series) -> pd.Series:
    return 1 - _normalize_pos(series)


def _missing_doc_list(row: pd.Series) -> List[str]:
    missing = []
    for doc in DOC_FIELDS:
        flag_col = f"doc_{doc}"
        if flag_col in row.index and not bool(row[flag_col]):
            missing.append(doc)
    return missing


def _suggest_product(row: pd.Series) -> str:
    if row["green_impact_score"] >= 78 and row["financial_resilience_score"] >= 66 and row["documentation_score"] >= 70:
        return "綠色設備貸款"
    if row["governance_score"] >= 70 and row["transition_readiness_score"] >= 72:
        return "永續績效連結貸款"
    if row["documentation_score"] < 60:
        return "先補件＋顧問輔導方案"
    if row["green_impact_score"] >= 65:
        return "轉型資金＋分階段授信"
    return "數據建置＋綠色授信培育方案"


def _green_fit_band(score: float) -> str:
    if score >= 80:
        return "A｜可優先推進"
    if score >= 68:
        return "B｜可推進但需補件"
    if score >= 55:
        return "C｜建議先輔導"
    return "D｜暫不建議送件"


def _red_flags(row: pd.Series) -> List[str]:
    flags = []
    if row["pd_estimate"] >= 0.09:
        flags.append("PD 偏高，需強化還款來源與保全條件")
    if row["environmental_controversy"]:
        flags.append("存在環境爭議紀錄，需補充改善說明")
    if row["climate_risk_score"] >= 0.70:
        flags.append("氣候實體/轉型風險較高")
    if row["customer_concentration"] >= 0.60:
        flags.append("客戶集中度偏高")
    if row["documentation_score"] < 60:
        flags.append("缺件較多，前審不宜直接推進")
    if not flags:
        flags.append("目前未見重大紅旗，仍建議依常規前審")
    return flags


def _post_loan_kpis(row: pd.Series) -> List[str]:
    return [
        f"月用電年減目標：{row['expected_energy_saving_pct']:.0%}",
        f"年度減碳目標：{row['expected_emission_reduction_tons']:,.0f} 噸 CO2e",
        "專案里程碑達成率",
        "再生能源占比變化",
        "主要客戶 ESG 問卷回覆完成率",
        "設備上線與驗收進度",
    ]


def build_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    master = datasets["master"].copy()
    annual = datasets["financials_annual"].copy()
    monthly = datasets["monthly_panel"].copy()
    esg = datasets["esg_documents"].copy()
    banking = datasets["banking_metrics"].copy()

    latest_fin = annual[annual["fiscal_year"] == annual["fiscal_year"].max()].copy()
    revenue_stats = monthly.groupby("company_id")["revenue_twd_mn"].agg(["mean", "std"]).reset_index()
    revenue_stats["revenue_cv"] = revenue_stats["std"] / revenue_stats["mean"].replace(0, np.nan)

    monthly_rollup = monthly.groupby("company_id").agg(
        annual_kwh=("electricity_kwh", "sum"),
        annual_power_cost_twd_mn=("electricity_cost_twd_mn", "sum"),
        renewable_kwh=("renewable_kwh", "sum"),
        avg_invoice_count=("invoice_count", "mean"),
    ).reset_index()
    monthly_rollup["renewable_share_calc"] = monthly_rollup["renewable_kwh"] / monthly_rollup["annual_kwh"].replace(0, np.nan)
    monthly_rollup["energy_intensity_kwh_per_twd_mn"] = monthly_rollup["annual_kwh"] / (
        monthly.groupby("company_id")["revenue_twd_mn"].sum().reindex(monthly_rollup["company_id"]).values
    )

    feat = (
        master
        .merge(latest_fin, on="company_id", how="left")
        .merge(revenue_stats[["company_id", "revenue_cv"]], on="company_id", how="left")
        .merge(monthly_rollup, on="company_id", how="left")
        .merge(esg, on="company_id", how="left")
        .merge(banking, on="company_id", how="left")
    )

    doc_cols = [c for c in feat.columns if c.startswith("doc_")]
    feat["documentation_score"] = feat[doc_cols].astype(float).mean(axis=1) * 100

    feat["financial_resilience_score"] = (
        0.24 * _normalize_pos(feat["interest_coverage"])
        + 0.20 * _normalize_neg(feat["debt_to_ebitda"])
        + 0.18 * _normalize_pos(feat["current_ratio"])
        + 0.18 * _normalize_pos(feat["dscr"])
        + 0.20 * _normalize_neg(feat["revenue_cv"].fillna(feat["revenue_cv"].median()))
    ) * 100

    feat["green_impact_score"] = (
        0.28 * _normalize_pos(feat["expected_energy_saving_pct"])
        + 0.24 * _normalize_pos(feat["expected_emission_reduction_tons"])
        + 0.16 * _normalize_neg(feat["payback_years"])
        + 0.16 * _normalize_pos(feat["renewable_share"])
        + 0.16 * _normalize_neg(feat["energy_intensity_kwh_per_twd_mn"])
    ) * 100

    feat["governance_score"] = (
        0.20 * feat["board_esg_oversight"].astype(float)
        + 0.18 * feat["iso_14001"].astype(float)
        + 0.14 * feat["iso_50001"].astype(float)
        + 0.14 * feat["iso_14064"].astype(float)
        + 0.16 * feat["carbon_inventory"].astype(float)
        + 0.18 * _normalize_pos(feat["governance_maturity"])
    ) * 100

    feat["transition_readiness_score"] = (
        0.24 * _normalize_pos(feat["digital_maturity"])
        + 0.18 * _normalize_pos(feat["documentation_completeness"])
        + 0.16 * _normalize_pos(feat["banking_readiness"])
        + 0.18 * _normalize_pos(feat["supply_chain_pressure"])
        + 0.24 * _normalize_pos(feat["governance_maturity"])
    ) * 100

    pd_latent = (
        -2.05
        - 0.018 * feat["financial_resilience_score"]
        - 0.010 * feat["transition_readiness_score"]
        - 0.010 * feat["governance_score"]
        + 0.16 * feat["debt_to_ebitda"]
        - 0.05 * feat["interest_coverage"]
        + 0.62 * feat["environmental_controversy"].astype(float)
        + 0.68 * feat["climate_risk_score"]
        + 0.48 * feat["customer_concentration"]
    )
    feat["pd_estimate"] = 1 / (1 + np.exp(-pd_latent))

    feat["panda_score"] = (
        0.28 * feat["financial_resilience_score"]
        + 0.24 * feat["green_impact_score"]
        + 0.18 * feat["governance_score"]
        + 0.16 * feat["transition_readiness_score"]
        + 0.14 * feat["documentation_score"]
    )

    feat["green_loan_fit_score"] = (
        0.45 * feat["panda_score"]
        + 0.20 * feat["banking_readiness"] * 100
        + 0.20 * (100 - feat["pd_estimate"] * 100)
        + 0.15 * feat["green_impact_score"]
    )

    feat["recommended_product"] = feat.apply(_suggest_product, axis=1)
    feat["green_loan_fit"] = feat["green_loan_fit_score"].apply(_green_fit_band)
    feat["missing_documents"] = feat.apply(_missing_doc_list, axis=1)
    feat["bank_red_flags"] = feat.apply(_red_flags, axis=1)
    feat["post_loan_kpis"] = feat.apply(_post_loan_kpis, axis=1)
    feat["carbon_fee_exposure_twd_mn"] = (
        feat["scope1_2_tons"] * CONFIG.default_carbon_price_twd_per_ton / 1_000_000
    )

    feat["enterprise_actions"] = feat["missing_documents"].apply(
        lambda docs: ["完成缺件上傳", "補充專案 KPI", "確認貸款用途與金流對應"] if docs else ["進入銀行前審", "確認授信額度", "安排專案追蹤"]
    )
    feat["bank_actions"] = feat["green_loan_fit"].apply(
        lambda x: ["啟動前審", "安排產業比對", "擬定授信條件"] if x.startswith("A")
        else ["要求補件", "確認還款來源", "補強 ESG/碳資料"] if x.startswith("B")
        else ["先做輔導", "建立資料模板", "安排顧問/設備商合作"]
    )
    return feat


def build_industry_benchmarks(scored: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "panda_score",
        "green_loan_fit_score",
        "financial_resilience_score",
        "green_impact_score",
        "governance_score",
        "documentation_score",
        "pd_estimate",
        "renewable_share",
        "energy_intensity_kwh_per_twd_mn",
    ]
    rows = []
    for industry, group in scored.groupby("industry"):
        for metric in metrics:
            rows.append(
                {
                    "industry": industry,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "median": group[metric].median(),
                    "p25": group[metric].quantile(0.25),
                    "p75": group[metric].quantile(0.75),
                    "count": len(group),
                }
            )
    return pd.DataFrame(rows)


def add_peer_comparisons(scored: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    key_metrics = ["panda_score", "green_loan_fit_score", "documentation_score", "green_impact_score", "pd_estimate"]
    for metric in key_metrics:
        mean_map = benchmarks[benchmarks["metric"] == metric].set_index("industry")["mean"].to_dict()
        out[f"{metric}_industry_mean"] = out["industry"].map(mean_map)
        out[f"{metric}_vs_industry"] = out[metric] - out[f"{metric}_industry_mean"]
    return out


def save_outputs(scored: pd.DataFrame, benchmarks: pd.DataFrame) -> None:
    scored.to_csv(CONFIG.reports_dir / "portfolio_scored.csv", index=False, encoding="utf-8-sig")
    benchmarks.to_csv(CONFIG.reports_dir / "industry_benchmarks.csv", index=False, encoding="utf-8-sig")
    save_company_documents(scored)

    summary = {
        "n_companies": int(len(scored)),
        "avg_panda_score": float(scored["panda_score"].mean()),
        "avg_green_loan_fit_score": float(scored["green_loan_fit_score"].mean()),
        "products": scored["recommended_product"].value_counts().to_dict(),
    }
    (CONFIG.reports_dir / "portfolio_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    datasets = load_datasets()
    feat = build_features(datasets)
    benchmarks = build_industry_benchmarks(feat)
    scored = add_peer_comparisons(feat, benchmarks)
    save_outputs(scored, benchmarks)


if __name__ == "__main__":
    main()
