from __future__ import annotations

import ast
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics_engine import (
    REPORT_CATALOG,
    company_comparison,
    industry_comparison,
    portfolio_snapshot,
    top_n_by_metric,
)
from config import CONFIG
from copilot import OllamaClient, PandaCopilot
from data_simulator import main as generate_data
from scoring import (
    add_peer_comparisons,
    build_features,
    build_industry_benchmarks,
    load_datasets,
    save_outputs,
)

st.set_page_config(
    page_title=CONFIG.app_title,
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.main-header {
    padding: 1.1rem 1.35rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0b3d2e 0%, #145c43 48%, #1f7a59 100%);
    color: white;
    margin-bottom: 1rem;
}
.section-card {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem 1rem 0.8rem 1rem;
    background: white;
}
.badge-good {
    background:#E7F7EF;
    color:#0B6E4F;
    padding:0.24rem 0.62rem;
    border-radius:999px;
    font-weight:700;
}
.badge-mid {
    background:#FFF4D6;
    color:#9C6A00;
    padding:0.24rem 0.62rem;
    border-radius:999px;
    font-weight:700;
}
.badge-bad {
    background:#FDE8E8;
    color:#B42318;
    padding:0.24rem 0.62rem;
    border-radius:999px;
    font-weight:700;
}
.small-muted {
    color:#6b7280;
    font-size:0.92rem;
}
.status-box {
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:0.8rem 0.9rem;
    background:#fafafa;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LIST_COLUMNS = [
    "missing_documents",
    "bank_red_flags",
    "post_loan_kpis",
    "bank_actions",
    "enterprise_actions",
]


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        for sep in ["；", ";", "、", ",", "\n"]:
            if sep in text:
                items = [x.strip(" \t\r\n'\"[]") for x in text.split(sep)]
                items = [x for x in items if x and len(x) > 0]
                if items:
                    return items

        return [text]

    return [str(value)]


def _normalize_scored_df(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    for col in LIST_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(_ensure_list)
    return out


@st.cache_data(show_spinner=False)
def ensure_and_load():
    if not (CONFIG.data_dir / "company_master.csv").exists():
        generate_data()

    if not (CONFIG.reports_dir / "portfolio_scored.csv").exists():
        datasets = load_datasets()
        feat = build_features(datasets)
        bmk = build_industry_benchmarks(feat)
        scored = add_peer_comparisons(feat, bmk)
        save_outputs(scored, bmk)

    datasets = load_datasets()
    scored = pd.read_csv(CONFIG.reports_dir / "portfolio_scored.csv")
    scored = _normalize_scored_df(scored)
    benchmarks = pd.read_csv(CONFIG.reports_dir / "industry_benchmarks.csv")
    return datasets, scored, benchmarks


def fit_badge(label: str) -> str:
    if str(label).startswith("A"):
        css = "badge-good"
    elif str(label).startswith("B") or str(label).startswith("C"):
        css = "badge-mid"
    else:
        css = "badge-bad"
    return f"<span class='{css}'>{label}</span>"


def bullet_lines(items: list[str]) -> str:
    items = _ensure_list(items)
    if not items:
        return "- 無"
    return "\n".join([f"- {item}" for item in items])


def st_df(df: pd.DataFrame, hide_index: bool = True):
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def st_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def st_full_button(label: str, **kwargs) -> bool:
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        try:
            return st.button(label, use_container_width=True, **kwargs)
        except TypeError:
            return st.button(label, **kwargs)


def get_copilot(
    scored: pd.DataFrame,
    benchmarks: pd.DataFrame,
    enable_llm: bool,
    model_name: str,
) -> PandaCopilot:
    llm = (
        OllamaClient(
            CONFIG.ollama_base_url,
            model_name=model_name,
            timeout=CONFIG.llm_timeout_sec,
        )
        if enable_llm
        else None
    )
    return PandaCopilot(scored, benchmarks, CONFIG.docs_dir, llm_client=llm)


def render_header():
    st.markdown(
        f"""
        <div class="main-header">
            <h1 style="margin:0;">🐼 {CONFIG.app_title}</h1>
            <div style="opacity:0.94; margin-top:0.35rem;">
                {CONFIG.app_subtitle}｜面向銀行端與企業端的綠色授信協作平台
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(scored: pd.DataFrame):
    with st.sidebar:
        st.header("平台設定")
        enable_llm = st.toggle("啟用本地 Ollama / Qwen", value=True)
        model_name = st.text_input("模型名稱", value=CONFIG.default_model)
        st.caption("建議使用 qwen3.6:27b。平台已預設關閉 thinking，並偏向完整、詳盡、全面回答。")

        if enable_llm:
            checker = OllamaClient(CONFIG.ollama_base_url, model_name=model_name)
            ok, message = checker.check_model_ready()
            ps_ok, ps_message = checker.ps()

            if ok:
                st.success(message)
            else:
                st.warning(message)

            if ps_ok:
                st.info(ps_message)
        else:
            st.info("目前已關閉 LLM，頁面會改用規則式內容。")

        industry_filter = st.multiselect("篩選產業", sorted(scored["industry"].unique()))
        fit_filter = st.multiselect("篩選適配等級", sorted(scored["green_loan_fit"].unique()))
        search = st.text_input("搜尋公司名稱 / 代碼")

        st.divider()
        st.caption("Panda Copilot 為第三方新創平台，協助企業送件、銀行前審、缺件管理、報告生成與貸後追蹤。")

    data = scored.copy()

    if industry_filter:
        data = data[data["industry"].isin(industry_filter)]

    if fit_filter:
        data = data[data["green_loan_fit"].isin(fit_filter)]

    if search:
        data = data[
            data["company_name"].str.contains(search, case=False, na=False)
            | data["company_id"].str.contains(search, case=False, na=False)
        ]

    return data, enable_llm, model_name


def render_overview(scored: pd.DataFrame):
    st.markdown("## 平台總覽")
    snap = portfolio_snapshot(scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平台企業數", snap["n_companies"])
    c2.metric("平均 Panda Score", f"{snap['avg_panda_score']:.1f}")
    c3.metric("平均 Green Loan Fit", f"{snap['avg_green_loan_fit_score']:.1f}")
    c4.metric("A 級可推進家數", snap["fit_A_count"])

    left, right = st.columns([1.3, 1.0])

    with left:
        fig = px.scatter(
            scored,
            x="green_impact_score",
            y="financial_resilience_score",
            color="industry",
            size="project_capex_twd_mn",
            hover_name="company_name",
            hover_data=["company_id", "recommended_product", "green_loan_fit"],
            title="企業分布：綠色效益 vs 財務韌性",
        )
        fig.update_layout(height=430)
        st_plot(fig)

    with right:
        product_dist = (
            scored["recommended_product"]
            .value_counts()
            .rename_axis("product")
            .reset_index(name="count")
        )
        fig2 = px.pie(product_dist, names="product", values="count", title="推薦產品分布")
        fig2.update_layout(height=430)
        st_plot(fig2)

    b1, b2 = st.columns([1.0, 1.0])

    with b1:
        st.markdown("### 最值得優先推進的案件")
        top_df = top_n_by_metric(scored, "green_loan_fit_score", ascending=False, n=10)[
            [
                "company_id",
                "company_name",
                "industry",
                "green_loan_fit_score",
                "recommended_product",
                "green_loan_fit",
            ]
        ]
        st_df(top_df)

    with b2:
        st.markdown("### 高缺件 / 高風險提醒")
        warn_df = scored.sort_values(
            ["documentation_score", "pd_estimate"],
            ascending=[True, False],
        ).head(10).copy()

        warn_df["missing_documents"] = warn_df["missing_documents"].apply(
            lambda x: "；".join(_ensure_list(x)[:3]) + ("…" if len(_ensure_list(x)) > 3 else "")
        )

        warn_df = warn_df[
            [
                "company_id",
                "company_name",
                "documentation_score",
                "pd_estimate",
                "missing_documents",
            ]
        ]
        st_df(warn_df)


def render_company_workspace(datasets: dict, source_df: pd.DataFrame, copilot: PandaCopilot):
    st.markdown("## 企業 / 銀行雙面工作區")

    cid = st.selectbox(
        "選擇公司",
        source_df["company_id"].tolist(),
        format_func=lambda x: f"{x}｜{source_df.loc[source_df['company_id'] == x, 'company_name'].iloc[0]}",
    )

    row = source_df[source_df["company_id"] == cid].iloc[0]
    monthly = datasets["monthly_panel"][datasets["monthly_panel"]["company_id"] == cid].copy()
    annual = datasets["financials_annual"][datasets["financials_annual"]["company_id"] == cid].copy()

    st.markdown(
        f"""
        <div class="section-card">
            <h3 style="margin-top:0;">{row['company_name']} <span class="small-muted">({row['company_id']})</span></h3>
            <div class="small-muted">{row['industry']} ｜ {row['region']} ｜ {row['supply_chain_role']} ｜ {row['relationship_stage']}</div>
            <div style="margin-top:0.6rem;">{row['description']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Panda Score", f"{row['panda_score']:.1f}")
    k2.metric("Green Loan Fit", f"{row['green_loan_fit_score']:.1f}")
    k3.metric("綠色效益分數", f"{row['green_impact_score']:.1f}")
    k4.metric("文件完整度", f"{row['documentation_score']:.1f}")
    k5.markdown(fit_badge(row["green_loan_fit"]), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["財務與營運", "綠色授信重點", "企業端摘要", "銀行前審 Memo"])

    with tab1:
        left, right = st.columns([1.3, 1.0])

        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["revenue_twd_mn"], name="月營收（百萬 TWD）"))
            fig.add_trace(
                go.Scatter(
                    x=monthly["date"],
                    y=monthly["electricity_kwh"] / 1000,
                    name="月用電（千 kWh）",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title="月營收與月用電趨勢",
                yaxis=dict(title="月營收（百萬 TWD）"),
                yaxis2=dict(title="月用電（千 kWh）", overlaying="y", side="right"),
                height=430,
            )
            st_plot(fig)

        with right:
            st.markdown("#### 近三年財務摘要")
            st_df(
                annual[
                    [
                        "fiscal_year",
                        "revenue_twd_mn",
                        "ebitda_margin_pct",
                        "debt_to_ebitda",
                        "interest_coverage",
                        "current_ratio",
                        "dscr",
                    ]
                ]
            )

    with tab2:
        left, right = st.columns([1.0, 1.0])

        with left:
            info = pd.DataFrame(
                {
                    "欄位": [
                        "主力產品建議",
                        "專案主題",
                        "預估 CAPEX（百萬 TWD）",
                        "預期節能",
                        "預期減碳（噸）",
                        "再生能源占比",
                        "年度 Scope 1+2（噸）",
                        "碳費暴露（百萬 TWD）",
                        "PD",
                        "主要紅旗",
                    ],
                    "值": [
                        row["recommended_product"],
                        row["transition_project"],
                        f"{row['project_capex_twd_mn']:.1f}",
                        f"{row['expected_energy_saving_pct']:.0%}",
                        f"{row['expected_emission_reduction_tons']:,.0f}",
                        f"{row['renewable_share']:.0%}",
                        f"{row['scope1_2_tons']:,.0f}",
                        f"{row['carbon_fee_exposure_twd_mn']:.2f}",
                        f"{row['pd_estimate']:.2%}",
                        "；".join(_ensure_list(row["bank_red_flags"])),
                    ],
                }
            )
            st_df(info)

        with right:
            st.markdown("#### 缺件清單")
            missing = _ensure_list(row["missing_documents"])
            if missing:
                st.markdown(bullet_lines(missing))
            else:
                st.success("目前未見重大缺件。")

            st.markdown("#### 貸後追蹤 KPI")
            st.markdown(bullet_lines(_ensure_list(row["post_loan_kpis"])))

    with tab3:
        enterprise_text, used_llm, message = copilot.build_enterprise_brief(cid)
        if used_llm:
            st.success(message)
        else:
            st.info(message)
        st.markdown(enterprise_text)

    with tab4:
        memo_text, used_llm, message = copilot.build_bank_memo(cid)
        if used_llm:
            st.success(message)
        else:
            st.info(message)
        st.markdown(memo_text)


def render_comparison(scored: pd.DataFrame, benchmarks: pd.DataFrame, copilot: PandaCopilot):
    st.markdown("## 比較與 Benchmark")

    selected = st.multiselect(
        "選擇兩家公司比較",
        scored["company_id"].tolist(),
        default=scored["company_id"].head(2).tolist(),
    )

    if len(selected) < 2:
        st.info("請至少選擇兩家公司。")
        return

    selected = selected[:2]
    cmp_df = company_comparison(scored, selected)
    st_df(cmp_df)

    radar = cmp_df.melt(
        id_vars=["company_id"],
        value_vars=[
            "financial_resilience_score",
            "green_impact_score",
            "governance_score",
            "documentation_score",
            "green_loan_fit_score",
        ],
        var_name="metric",
        value_name="score",
    )

    fig = px.line_polar(
        radar,
        r="score",
        theta="metric",
        color="company_id",
        line_close=True,
        title="五大核心面向比較",
    )
    fig.update_layout(height=460)
    st_plot(fig)

    st.markdown("### 自動比較摘要")
    compare_result = copilot.answer_question(
        f"比較 {selected[0]} 和 {selected[1]} 的綠色授信適配、風險、缺件與建議產品"
    )
    if compare_result.get("used_llm"):
        st.success(compare_result.get("message", "已使用 LLM。"))
    else:
        st.info(compare_result.get("message", "目前使用規則式輸出。"))
    st.markdown(compare_result["answer"])

    st.markdown("### 同產業比較")
    cid = st.selectbox("選擇要看同產業比較的公司", scored["company_id"].tolist(), key="industry_cmp")
    result = industry_comparison(scored, benchmarks, cid)
    company = result["company"]
    industry_mean = result["industry_mean"]

    compare_text = (
        f"{company['company_name']} 所在產業為 {company['industry']}。"
        f" Panda Score：{company['panda_score']:.1f}（產業平均 {industry_mean.get('panda_score', float('nan')):.1f}），"
        f" 文件完整度：{company['documentation_score']:.1f}（產業平均 {industry_mean.get('documentation_score', float('nan')):.1f}）。"
    )
    st.info(compare_text)


def render_report_studio(scored: pd.DataFrame, copilot: PandaCopilot):
    st.markdown("## Report Studio")

    cid = st.selectbox("選擇公司", scored["company_id"].tolist(), key="report_company")
    report_type = st.selectbox("選擇生成內容", REPORT_CATALOG)

    report_text, used_llm, message = copilot.build_report(cid, report_type)
    if used_llm:
        st.success(message)
    else:
        st.info(message)
    st.markdown(report_text)

    st.markdown("### Panda Copilot 可生成的內容")
    extra = [
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
    st.markdown(bullet_lines(extra))


def render_copilot_console(scored: pd.DataFrame, copilot: PandaCopilot):
    st.markdown("## Copilot Console")

    quick_tasks = [
        "CMP_001 最適合哪類金融產品？請完整分析原因、風險、缺件與下一步。",
        "CMP_001 還缺哪些文件？請說明對授信流程的影響。",
        "請為 CMP_001 生成完整銀行前審 Memo",
        f"比較 {scored['company_id'].iloc[0]} 和 {scored['company_id'].iloc[1]}，請完整分析誰比較適合先推進。",
    ]

    task = st.selectbox("快速任務", quick_tasks)
    custom = st.text_area("或自行輸入問題", value=task, height=120)

    if st_full_button("執行問題分析"):
        result = copilot.answer_question(custom)
        if result.get("used_llm"):
            st.success(result.get("message", "本次回答已使用 Ollama / Qwen。"))
        else:
            st.info(result.get("message", "本次回答使用平台規則式輸出。"))
        st.markdown(result["answer"])


def main():
    datasets, scored, benchmarks = ensure_and_load()
    filtered, enable_llm, model_name = render_sidebar(scored)
    copilot = get_copilot(scored, benchmarks, enable_llm, model_name)

    render_header()

    workspace_df = filtered if not filtered.empty else scored
    pages = st.tabs(["平台總覽", "雙面工作區", "比較與 Benchmark", "Report Studio", "Copilot Console"])

    with pages[0]:
        render_overview(workspace_df)

    with pages[1]:
        render_company_workspace(datasets, workspace_df, copilot)

    with pages[2]:
        compare_df = workspace_df if len(workspace_df) >= 2 else scored
        render_comparison(compare_df, benchmarks, copilot)

    with pages[3]:
        render_report_studio(workspace_df, copilot)

    with pages[4]:
        render_copilot_console(workspace_df, copilot)


if __name__ == "__main__":
    main()