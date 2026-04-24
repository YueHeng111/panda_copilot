from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG


INDUSTRY_PROFILES = {
    "食品加工": {
        "regions": ["桃園", "台中", "高雄", "彰化"],
        "energy_intensity": (620, 980),
        "revenue_mn": (180, 2600),
        "ebitda_margin": (0.09, 0.18),
        "debt_to_ebitda": (1.2, 4.5),
        "renewable_share": (0.04, 0.20),
        "project_types": ["冷凍設備汰換", "高效鍋爐", "屋頂光電", "熱回收"],
        "top_pressure": ["零售通路", "品牌商", "出口客戶"],
    },
    "冷鏈物流": {
        "regions": ["新北", "桃園", "台中", "高雄"],
        "energy_intensity": (850, 1500),
        "revenue_mn": (150, 1800),
        "ebitda_margin": (0.07, 0.15),
        "debt_to_ebitda": (1.8, 5.0),
        "renewable_share": (0.03, 0.16),
        "project_types": ["冷媒改善", "車隊電動化", "能源管理系統", "冷凍機汰換"],
        "top_pressure": ["量販通路", "食品品牌", "電商平台"],
    },
    "金屬加工": {
        "regions": ["台中", "台南", "高雄", "彰化"],
        "energy_intensity": (700, 1350),
        "revenue_mn": (220, 3200),
        "ebitda_margin": (0.10, 0.20),
        "debt_to_ebitda": (1.0, 3.8),
        "renewable_share": (0.05, 0.22),
        "project_types": ["空壓機改善", "高效馬達", "熱處理優化", "製程數位監控"],
        "top_pressure": ["汽車供應鏈", "出口客戶", "品牌製造商"],
    },
    "電子零組件": {
        "regions": ["新竹", "桃園", "台中", "台南"],
        "energy_intensity": (680, 1450),
        "revenue_mn": (300, 5200),
        "ebitda_margin": (0.12, 0.25),
        "debt_to_ebitda": (0.8, 3.2),
        "renewable_share": (0.08, 0.35),
        "project_types": ["PPA 綠電採購", "空調優化", "製程節能", "ISO 50001 導入"],
        "top_pressure": ["國際品牌", "半導體客戶", "ESG 採購規範"],
    },
    "紡織成衣": {
        "regions": ["彰化", "台南", "高雄", "桃園"],
        "energy_intensity": (720, 1280),
        "revenue_mn": (160, 2100),
        "ebitda_margin": (0.07, 0.16),
        "debt_to_ebitda": (1.6, 4.8),
        "renewable_share": (0.03, 0.18),
        "project_types": ["染整節水", "蒸汽系統更新", "屋頂光電", "廢熱回收"],
        "top_pressure": ["國際服飾品牌", "出口客戶", "供應鏈稽核"],
    },
    "塑膠與包材": {
        "regions": ["新北", "桃園", "台中", "高雄"],
        "energy_intensity": (540, 1150),
        "revenue_mn": (180, 2600),
        "ebitda_margin": (0.08, 0.17),
        "debt_to_ebitda": (1.2, 4.2),
        "renewable_share": (0.04, 0.20),
        "project_types": ["射出設備汰換", "循環材料導入", "空壓節能", "能源監測"],
        "top_pressure": ["品牌客戶", "零售通路", "出口法規"],
    },
}

RELATIONSHIP_STAGES = ["第三方導入", "銀行邀請入平台", "已送件待補件", "前審中", "貸後追蹤中"]
COMPANY_ROLES = ["一階供應商", "二階供應商", "OEM/ODM", "通路供應商", "出口導向供應商"]
OWNERSHIP_TYPES = ["家族企業", "民營企業", "上市櫃企業", "外資背景"]
DOC_FIELDS = [
    "台電帳單",
    "電子發票",
    "財報摘要",
    "銀行往來摘要",
    "專案計畫書",
    "設備報價單",
    "碳盤查資料",
    "ISO 文件",
    "供應鏈問卷",
    "還款來源說明",
]
PREFIX = ["台", "永", "宏", "盛", "嘉", "鼎", "華", "富", "群", "信", "和"]
SUFFIX = ["實業", "科技", "工業", "材料", "能源", "物流", "精密", "股份"]


def _rand(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _pick_name(rng: np.random.Generator, industry: str, idx: int) -> str:
    hint = {
        "食品加工": "食品",
        "冷鏈物流": "冷鏈",
        "金屬加工": "精工",
        "電子零組件": "電子",
        "紡織成衣": "紡織",
        "塑膠與包材": "包材",
    }[industry]
    return f"{rng.choice(PREFIX)}{hint}{rng.choice(SUFFIX)}{idx:03d}"


def simulate_company_master(n_companies: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    industries = list(INDUSTRY_PROFILES.keys())
    for idx in range(1, n_companies + 1):
        industry = rng.choice(industries)
        profile = INDUSTRY_PROFILES[industry]
        revenue_mn = _rand(rng, *profile["revenue_mn"])
        employees = int(rng.integers(35, 1200))
        export_ratio = _rand(rng, 0.10, 0.92)
        digital_maturity = _rand(rng, 0.35, 0.96)
        supply_chain_pressure = _rand(rng, 0.35, 0.95)
        company_id = f"CMP_{idx:03d}"
        transition_project = rng.choice(profile["project_types"])
        relationship_stage = rng.choice(RELATIONSHIP_STAGES, p=[0.28, 0.20, 0.18, 0.22, 0.12])
        company = {
            "company_id": company_id,
            "company_name": _pick_name(rng, industry, idx),
            "industry": industry,
            "region": rng.choice(profile["regions"]),
            "ownership_type": rng.choice(OWNERSHIP_TYPES, p=[0.34, 0.33, 0.22, 0.11]),
            "supply_chain_role": rng.choice(COMPANY_ROLES),
            "relationship_stage": relationship_stage,
            "employees": employees,
            "revenue_twd_mn": revenue_mn,
            "export_ratio": export_ratio,
            "digital_maturity": digital_maturity,
            "supply_chain_pressure": supply_chain_pressure,
            "major_customer_pressure": rng.choice(profile["top_pressure"]),
            "transition_project": transition_project,
            "project_capex_twd_mn": _rand(rng, 15, 260),
            "expected_energy_saving_pct": _rand(rng, 0.06, 0.34),
            "expected_emission_reduction_tons": _rand(rng, 80, 5800),
            "payback_years": _rand(rng, 1.4, 6.8),
            "ownership_years": int(rng.integers(4, 42)),
            "customer_concentration": _rand(rng, 0.18, 0.82),
            "supplier_concentration": _rand(rng, 0.12, 0.74),
            "description": (
                f"{transition_project} 為主要投資主題，企業位於 {rng.choice(profile['regions']) if False else profile['regions'][0]}"
            ),
        }
        company["description"] = (
            f"{company['company_name']} 為 {industry} 企業，現階段主要關注「{transition_project}」專案，"
            f"同時面臨 {company['major_customer_pressure']} 帶來的減碳揭露與交期要求。"
        )
        rows.append(company)
    return pd.DataFrame(rows)


def simulate_financials(master: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _, row in master.iterrows():
        profile = INDUSTRY_PROFILES[row["industry"]]
        base_revenue = row["revenue_twd_mn"]
        base_margin = _rand(rng, *profile["ebitda_margin"])
        base_debt = _rand(rng, *profile["debt_to_ebitda"])
        current_ratio = _rand(rng, 0.9, 2.3)
        interest_coverage = _rand(rng, 1.1, 9.5)
        dscr = _rand(rng, 0.95, 2.8)
        for year in [2023, 2024, 2025]:
            growth = _rand(rng, -0.04, 0.18)
            revenue = base_revenue * ((1 + growth) ** (year - 2025)) * (1 + rng.normal(0, 0.04))
            ebitda_margin = float(np.clip(base_margin + rng.normal(0, 0.015), 0.04, 0.32))
            debt_to_ebitda = float(np.clip(base_debt + rng.normal(0, 0.25), 0.5, 6.2))
            op_cf = revenue * float(np.clip(_rand(rng, 0.05, 0.18), 0.03, 0.24))
            interest_cov = float(np.clip(interest_coverage + rng.normal(0, 0.6), 0.7, 12.0))
            dscr_value = float(np.clip(dscr + rng.normal(0, 0.15), 0.6, 4.0))
            rows.append(
                {
                    "company_id": row["company_id"],
                    "fiscal_year": year,
                    "revenue_twd_mn": revenue,
                    "ebitda_margin_pct": ebitda_margin * 100,
                    "debt_to_ebitda": debt_to_ebitda,
                    "interest_coverage": interest_cov,
                    "current_ratio": float(np.clip(current_ratio + rng.normal(0, 0.15), 0.65, 2.8)),
                    "dscr": dscr_value,
                    "operating_cashflow_twd_mn": op_cf,
                    "capex_twd_mn": row["project_capex_twd_mn"] * _rand(rng, 0.45, 1.20),
                }
            )
    return pd.DataFrame(rows)


def simulate_monthly_panel(master: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    rows = []
    seasonal = np.sin(np.linspace(0, 4 * math.pi, CONFIG.monthly_periods))
    for _, row in master.iterrows():
        profile = INDUSTRY_PROFILES[row["industry"]]
        annual_rev = row["revenue_twd_mn"]
        energy_intensity = _rand(rng, *profile["energy_intensity"])
        renewable_share = _rand(rng, *profile["renewable_share"])
        for i, dt in enumerate(CONFIG.monthly_dates):
            revenue = max(annual_rev / 12 * (1 + seasonal[i] * _rand(rng, 0.04, 0.15)) * (1 + rng.normal(0, 0.03)), 5.0)
            electricity_kwh = max(revenue * energy_intensity, 12000.0)
            tariff = _rand(rng, 2.65, 3.75)
            rows.append(
                {
                    "company_id": row["company_id"],
                    "date": dt,
                    "revenue_twd_mn": revenue,
                    "electricity_kwh": electricity_kwh,
                    "electricity_cost_twd_mn": electricity_kwh * tariff / 1_000_000,
                    "renewable_kwh": electricity_kwh * renewable_share,
                    "invoice_count": int(max(12, revenue * _rand(rng, 20, 50))),
                }
            )
    return pd.DataFrame(rows)


def simulate_esg_and_documents(master: pd.DataFrame, monthly: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    rows = []
    for _, row in master.iterrows():
        profile = INDUSTRY_PROFILES[row["industry"]]
        panel = monthly[monthly["company_id"] == row["company_id"]]
        annual_kwh = panel["electricity_kwh"].sum()
        renewable_share = float(np.clip(panel["renewable_kwh"].sum() / max(annual_kwh, 1), 0, 1))
        scope1_2 = annual_kwh * CONFIG.taiwan_grid_factor_kg_per_kwh / 1000
        controversy_flag = bool(rng.choice([False, True], p=[0.84, 0.16]))
        iso_14001 = bool(rng.choice([True, False], p=[0.72, 0.28]))
        iso_50001 = bool(rng.choice([True, False], p=[0.50, 0.50]))
        iso_14064 = bool(rng.choice([True, False], p=[0.43, 0.57]))
        board_esg = bool(rng.choice([True, False], p=[0.62, 0.38]))
        carbon_inventory = bool(rng.choice([True, False], p=[0.56, 0.44]))
        docs = {}
        for doc in DOC_FIELDS:
            base_prob = 0.68 if doc in {"台電帳單", "電子發票", "財報摘要"} else 0.52
            if doc in {"專案計畫書", "設備報價單"}:
                base_prob = 0.74
            if doc == "碳盤查資料":
                base_prob = 0.45
            if doc == "ISO 文件":
                base_prob = 0.50
            docs[f"doc_{doc}"] = bool(rng.choice([True, False], p=[base_prob, 1 - base_prob]))
        rows.append(
            {
                "company_id": row["company_id"],
                "renewable_share": renewable_share,
                "scope1_2_tons": scope1_2,
                "estimated_scope3_tons": scope1_2 * _rand(rng, 1.5, 6.5),
                "climate_risk_score": _rand(rng, 0.20, 0.86),
                "water_risk_score": _rand(rng, 0.14, 0.82),
                "governance_maturity": _rand(rng, 0.30, 0.94),
                "board_esg_oversight": board_esg,
                "iso_14001": iso_14001,
                "iso_50001": iso_50001,
                "iso_14064": iso_14064,
                "carbon_inventory": carbon_inventory,
                "net_zero_commitment": bool(rng.choice([True, False], p=[0.58, 0.42])),
                "environmental_controversy": controversy_flag,
                "esg_gap_summary": rng.choice(
                    [
                        "再生能源採購路徑不明確",
                        "碳盤查邊界與頻率不足",
                        "供應鏈問卷回覆資料分散",
                        "節能 KPI 缺乏月度追蹤",
                        "董事會永續治理尚未制度化",
                    ]
                ),
                "transition_barrier": rng.choice(
                    [
                        "CAPEX 壓力偏高",
                        "綠電取得成本與合約不確定",
                        "內部資料治理成熟度不足",
                        "設備汰換與產能安排衝突",
                        "客戶要求加速但專案排程落後",
                    ]
                ),
                "top_expected_benefit": rng.choice(
                    ["降低能耗成本", "滿足品牌供應鏈要求", "改善碳費暴露", "提升貸款可核貸性", "爭取新訂單資格"]
                ),
                **docs,
            }
        )
    return pd.DataFrame(rows)


def simulate_banking(master: pd.DataFrame, financials: pd.DataFrame, esg_docs: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4)
    latest = financials[financials["fiscal_year"] == 2025].copy()
    rows = []
    for _, row in master.iterrows():
        fin = latest[latest["company_id"] == row["company_id"]].iloc[0]
        docs = esg_docs[esg_docs["company_id"] == row["company_id"]].iloc[0]
        docs_ready = np.mean([docs[c] for c in docs.index if c.startswith("doc_")])
        relationship_score = {
            "第三方導入": 0.46,
            "銀行邀請入平台": 0.62,
            "已送件待補件": 0.70,
            "前審中": 0.78,
            "貸後追蹤中": 0.86,
        }[row["relationship_stage"]]
        repayment_quality = np.clip(
            0.34 * (fin["operating_cashflow_twd_mn"] / max(fin["revenue_twd_mn"], 1))
            + 0.18 * min(fin["dscr"] / 2.2, 1.0)
            + 0.14 * min(fin["interest_coverage"] / 6.0, 1.0)
            + 0.14 * (1 - min(row["customer_concentration"], 1))
            + 0.20 * relationship_score,
            0,
            1,
        )
        rows.append(
            {
                "company_id": row["company_id"],
                "relationship_score": relationship_score,
                "documentation_completeness": docs_ready,
                "repayment_quality": repayment_quality,
                "banking_readiness": np.clip(
                    0.30 * relationship_score + 0.35 * docs_ready + 0.35 * repayment_quality,
                    0,
                    1,
                ),
                "requested_loan_twd_mn": row["project_capex_twd_mn"] * _rand(rng, 0.65, 1.25),
                "collateral_strength": _rand(rng, 0.20, 0.95),
                "loan_purpose_note": f"用於 {row['transition_project']} 與相關節能轉型投資",
                "repayment_source_note": rng.choice(
                    [
                        "主要來自穩定訂單與營運現金流",
                        "主要來自長約客戶應收與產能提升後新增收入",
                        "主要來自節能節費效益與既有客戶續單",
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def save_company_documents(dataset: pd.DataFrame) -> None:
    docs_root = CONFIG.docs_dir
    for _, row in dataset.iterrows():
        company_dir = docs_root / row["company_id"]
        company_dir.mkdir(parents=True, exist_ok=True)

        profile_text = f"""# {row['company_name']} 公司簡介
公司代碼：{row['company_id']}
產業：{row['industry']}
區域：{row['region']}
供應鏈角色：{row['supply_chain_role']}
平台關係階段：{row['relationship_stage']}
專案主題：{row['transition_project']}

企業描述：
{row['description']}

主要壓力來源：
- 客戶壓力：{row['major_customer_pressure']}
- ESG 缺口：{row['esg_gap_summary']}
- 轉型障礙：{row['transition_barrier']}
- 預期效益：{row['top_expected_benefit']}
"""
        (company_dir / "company_profile.md").write_text(profile_text, encoding="utf-8")

        bank_context = {
            "company_id": row["company_id"],
            "company_name": row["company_name"],
            "requested_loan_twd_mn": row["requested_loan_twd_mn"],
            "loan_purpose_note": row["loan_purpose_note"],
            "repayment_source_note": row["repayment_source_note"],
            "recommended_product": row["recommended_product"],
            "green_loan_fit": row["green_loan_fit"],
            "missing_documents": row["missing_documents"],
            "bank_red_flags": row["bank_red_flags"],
            "post_loan_kpis": row["post_loan_kpis"],
        }
        (company_dir / "bank_context.json").write_text(
            json.dumps(bank_context, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    master = simulate_company_master(CONFIG.n_companies, CONFIG.random_seed)
    financials = simulate_financials(master, CONFIG.random_seed)
    monthly = simulate_monthly_panel(master, CONFIG.random_seed)
    esg_docs = simulate_esg_and_documents(master, monthly, CONFIG.random_seed)
    banking = simulate_banking(master, financials, esg_docs, CONFIG.random_seed)

    master.to_csv(CONFIG.data_dir / "company_master.csv", index=False, encoding="utf-8-sig")
    financials.to_csv(CONFIG.data_dir / "financials_annual.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(CONFIG.data_dir / "monthly_panel.csv", index=False, encoding="utf-8-sig")
    esg_docs.to_csv(CONFIG.data_dir / "esg_documents.csv", index=False, encoding="utf-8-sig")
    banking.to_csv(CONFIG.data_dir / "banking_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
