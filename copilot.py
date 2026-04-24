from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from analytics_engine import industry_comparison
from config import CONFIG
from parsers import load_company_documents


@dataclass
class LLMResponse:
    text: str
    used_llm: bool
    model_name: str = ""
    error: Optional[str] = None


class OllamaClient:
    def __init__(self, base_url: str, model_name: str, timeout: int = 240):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def _get(self, path: str, timeout: int = 10) -> requests.Response:
        resp = requests.get(f"{self.base_url}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp

    def check_model_ready(self) -> tuple[bool, str]:
        try:
            tags_resp = self._get("/api/tags")
            models = tags_resp.json().get("models", [])
            names = {m.get("name", "") for m in models}
            if self.model_name in names:
                return True, f"{self.model_name} 已可使用"
            same_family = sorted(n for n in names if n.split(":")[0] == self.model_name.split(":")[0])
            if same_family:
                return False, f"找不到 {self.model_name}，但已安裝：{', '.join(same_family)}"
            return False, f"找不到模型 {self.model_name}，請先執行 ollama pull {self.model_name}"
        except Exception as exc:
            return False, f"無法連線到 Ollama：{exc}"

    def ps(self) -> tuple[bool, str]:
        try:
            resp = self._get("/api/ps")
            models = resp.json().get("models", [])
            if not models:
                return True, "目前沒有模型常駐記憶體，首次呼叫時會自動載入。"
            running = ", ".join(m.get("name", "") for m in models)
            return True, f"目前常駐：{running}"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _clean_model_text(text: str) -> str:
        if not text:
            return ""
        cleaned = text
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S | re.I)
        cleaned = re.sub(r"^Thinking\.\.\.[\s\S]*?\.\.\.done thinking\.\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"^Thinking Process:[\s\S]*?(?=Final check|最終回答|答案：)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "keep_alive": CONFIG.llm_keep_alive,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": CONFIG.llm_temperature,
                "num_ctx": CONFIG.llm_num_ctx,
                "num_predict": CONFIG.llm_num_predict,
                "top_p": 0.90,
                "repeat_penalty": 1.05,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "") or ""
            text = self._clean_model_text(text)
            if not text:
                return LLMResponse(text="", used_llm=False, model_name=self.model_name, error="模型未回傳有效內容")
            return LLMResponse(text=text, used_llm=True, model_name=self.model_name)
        except Exception as exc:
            return LLMResponse(text="", used_llm=False, model_name=self.model_name, error=str(exc))


def token_overlap_score(query: str, text: str) -> float:
    q = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    t = set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
    if not q:
        return 0.0
    return len(q & t) / math.sqrt(max(len(q), 1) * max(len(t), 1))


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
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
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        separators = ["；", ";", "、", ",", "\n"]
        for sep in separators:
            if sep in text:
                items = [x.strip(" \t\r\n'\"[]") for x in text.split(sep)]
                items = [x for x in items if x and len(x) > 1]
                if items:
                    return items
        return [text]
    return [str(value)]


def _bullet_markdown(items: List[str]) -> str:
    items = _ensure_list(items)
    if not items:
        return "- 無"
    return "\n".join(f"- {item}" for item in items)


class PandaCopilot:
    LIST_COLUMNS = ["missing_documents", "bank_red_flags", "post_loan_kpis", "bank_actions", "enterprise_actions"]

    def __init__(
        self,
        scored: pd.DataFrame,
        benchmarks: pd.DataFrame,
        docs_root: Path,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.scored = scored.copy()
        for col in self.LIST_COLUMNS:
            if col in self.scored.columns:
                self.scored[col] = self.scored[col].apply(_ensure_list)
        self.benchmarks = benchmarks.copy()
        self.docs_root = docs_root
        self.llm = llm_client
        self.scored["company_id"] = self.scored["company_id"].astype(str)

    def _llm_ready(self) -> bool:
        return self.llm is not None

    def get_company(self, company_id: str) -> pd.Series:
        return self.scored[self.scored["company_id"] == company_id].iloc[0]

    def _row_context(self, row: pd.Series) -> Dict[str, object]:
        keep = [
            "company_id", "company_name", "industry", "region", "supply_chain_role", "relationship_stage",
            "transition_project", "project_capex_twd_mn", "expected_energy_saving_pct", "expected_emission_reduction_tons",
            "payback_years", "revenue_twd_mn", "ebitda_margin_pct", "debt_to_ebitda", "interest_coverage",
            "current_ratio", "dscr", "annual_kwh", "annual_power_cost_twd_mn", "renewable_share", "scope1_2_tons",
            "carbon_fee_exposure_twd_mn", "customer_concentration", "supply_chain_pressure", "climate_risk_score",
            "governance_score", "documentation_score", "financial_resilience_score", "green_impact_score",
            "transition_readiness_score", "panda_score", "green_loan_fit_score", "green_loan_fit", "pd_estimate",
            "recommended_product", "missing_documents", "bank_red_flags", "post_loan_kpis", "bank_actions",
            "enterprise_actions", "loan_purpose_note", "repayment_source_note", "esg_gap_summary", "transition_barrier",
            "top_expected_benefit", "description",
        ]
        payload: Dict[str, object] = {}
        for key in keep:
            if key in row.index:
                value = row[key]
                if isinstance(value, np.generic):
                    value = value.item()
                if key in self.LIST_COLUMNS:
                    value = _ensure_list(value)
                payload[key] = value
        return payload

    def _retrieve_docs(self, company_ids: List[str], query: str, top_k: int = 8) -> str:
        docs = []
        for cid in company_ids:
            chunks = load_company_documents(self.docs_root / cid, cid)
            for chunk in chunks:
                score = token_overlap_score(query, f"{chunk.title} {chunk.text}")
                docs.append((score, chunk))
        docs.sort(key=lambda item: item[0], reverse=True)
        selected = docs[:top_k]
        return "\n\n".join(
            f"[{chunk.company_id} | {chunk.source_name} | {chunk.title}]\n{chunk.text}" for _, chunk in selected
        )

    def _llm_or_fallback(self, system_prompt: str, user_prompt: str, fallback: str) -> tuple[str, bool, str]:
        if not self._llm_ready():
            return fallback, False, "目前未啟用 LLM，已使用平台規則式內容。"
        res = self.llm.generate(system_prompt, user_prompt)
        if res.used_llm and res.text:
            return res.text, True, f"已使用 {res.model_name} 生成內容。"
        message = res.error or "LLM 暫時不可用，已改用規則式內容。"
        return fallback, False, message

    def _enterprise_fallback(self, row: pd.Series) -> str:
        return f"""### 企業端送件摘要

**一、公司與專案概況**
- 公司：{row['company_name']}（{row['company_id']}）
- 產業／地區：{row['industry']}／{row['region']}
- 供應鏈角色：{row['supply_chain_role']}
- 本次專案：{row['transition_project']}
- 建議主力產品：{row['recommended_product']}
- 綠色授信適配等級：{row['green_loan_fit']}

**二、申請亮點**
- Panda Score：{row['panda_score']:.1f}
- Green Loan Fit Score：{row['green_loan_fit_score']:.1f}
- 預期節能：{row['expected_energy_saving_pct']:.0%}
- 預期減碳：{row['expected_emission_reduction_tons']:,.0f} 噸 CO2e
- 預估專案投資額：{row['project_capex_twd_mn']:.1f} 百萬 TWD
- 主要效益：{row['top_expected_benefit']}

**三、目前待補資料**
{_bullet_markdown(row['missing_documents'])}

**四、建議下一步**
{_bullet_markdown(row['enterprise_actions'])}
""".strip()

    def build_enterprise_brief(self, company_id: str) -> tuple[str, bool, str]:
        row = self.get_company(company_id)
        fallback = self._enterprise_fallback(row)
        docs = self._retrieve_docs([company_id], "企業送件摘要 補件 專案效益 還款來源 ESG")
        system_prompt = """
你是 Panda Copilot 的企業端綠色授信顧問，服務對象是準備申請綠色貸款或轉型融資的企業窗口。
請用繁體中文，輸出完整、詳細、專業、可直接提供給企業內部財務/永續/設備窗口使用的內容。
嚴格遵守：
1. 只能根據提供資料撰寫，不可虛構數字、政策、認證或文件。
2. 不要簡短帶過；每個段落都要有明確說明與實務建議。
3. 不要輸出 Python list、JSON、程式碼或 thinking 過程。
4. 格式請使用 Markdown，至少包含以下段落：
   - 一、公司與專案概況
   - 二、本案對銀行有吸引力的原因
   - 三、目前缺口與待補件重點
   - 四、建議企業下一步行動
   - 五、送件時應強調的說法
""".strip()
        user_prompt = f"請為下列公司生成完整企業送件摘要。\n\n結構化資料：\n{json.dumps(self._row_context(row), ensure_ascii=False, indent=2)}\n\n參考文件：\n{docs}"
        return self._llm_or_fallback(system_prompt, user_prompt, fallback)

    def _bank_memo_fallback(self, row: pd.Series) -> str:
        return f"""## Panda Copilot｜銀行前審 Memo

**一、案件結論**
- 公司：{row['company_name']}（{row['company_id']}）
- 本案建議產品：{row['recommended_product']}
- 綠色授信適配等級：{row['green_loan_fit']}
- Panda Score：{row['panda_score']:.1f}
- Green Loan Fit Score：{row['green_loan_fit_score']:.1f}

**二、授信與還款重點**
- 專案用途：{row['loan_purpose_note']}
- 還款來源：{row['repayment_source_note']}
- 財務韌性分數：{row['financial_resilience_score']:.1f}
- 文件完整度：{row['documentation_score']:.1f}
- PD 估計：{row['pd_estimate']:.2%}

**三、綠色效益與轉型重點**
- 專案：{row['transition_project']}
- 預估 CAPEX：{row['project_capex_twd_mn']:.1f} 百萬 TWD
- 預期節能：{row['expected_energy_saving_pct']:.0%}
- 預期減碳：{row['expected_emission_reduction_tons']:,.0f} 噸 CO2e
- 再生能源占比：{row['renewable_share']:.0%}
- 年度 Scope 1+2：{row['scope1_2_tons']:,.0f} 噸
- 碳費暴露：{row['carbon_fee_exposure_twd_mn']:.2f} 百萬 TWD

**四、主要紅旗**
{_bullet_markdown(row['bank_red_flags'])}

**五、待補件**
{_bullet_markdown(row['missing_documents'])}

**六、貸後建議追蹤 KPI**
{_bullet_markdown(row['post_loan_kpis'])}
""".strip()

    def build_bank_memo(self, company_id: str) -> tuple[str, bool, str]:
        row = self.get_company(company_id)
        fallback = self._bank_memo_fallback(row)
        docs = self._retrieve_docs([company_id], "銀行前審 memo 風險 補件 KPI 授信 還款來源")
        system_prompt = """
你是 Panda Copilot 的銀行端綠色授信前審助手，讀者是銀行 RM、授信審查、永續金融與風險管理人員。
請用繁體中文輸出完整、詳細、專業且實務可用的前審 Memo。
嚴格遵守：
1. 只能依據提供資料與文件內容撰寫，不可虛構事實。
2. 不要簡短摘要，要完整說明案件判斷依據。
3. 格式請使用 Markdown，至少包含：
   - 一、案件結論與建議產品
   - 二、授信判斷依據
   - 三、綠色效益與專案合理性
   - 四、風險與紅旗
   - 五、待補件清單
   - 六、建議條件與後續行動
   - 七、貸後追蹤 KPI
4. 請避免輸出 JSON、list 原始字串與 thinking 過程。
""".strip()
        user_prompt = f"請為下列公司生成完整銀行前審 Memo。\n\n結構化資料：\n{json.dumps(self._row_context(row), ensure_ascii=False, indent=2)}\n\n參考文件：\n{docs}"
        return self._llm_or_fallback(system_prompt, user_prompt, fallback)

    def build_missing_doc_report(self, company_id: str) -> str:
        row = self.get_company(company_id)
        return f"""### 缺件與補件清單

**案件：** {row['company_name']}（{row['company_id']}）

**仍需補件**
{_bullet_markdown(row['missing_documents'])}

**銀行端建議下一步**
{_bullet_markdown(row['bank_actions'])}

**企業端建議下一步**
{_bullet_markdown(row['enterprise_actions'])}
""".strip()

    def build_report(self, company_id: str, report_type: str) -> tuple[str, bool, str]:
        row = self.get_company(company_id)

        fallback_map = {
            "企業綠色授信申請摘要": self._enterprise_fallback(row),
            "銀行前審 Memo": self._bank_memo_fallback(row),
            "缺件與補件清單": self.build_missing_doc_report(company_id),
            "綠色貸款適配建議": f"""### 綠色貸款適配建議

- 公司：{row['company_name']}（{row['company_id']}）
- 適配等級：{row['green_loan_fit']}
- 建議主力產品：{row['recommended_product']}
- 判斷依據：Panda Score {row['panda_score']:.1f}、文件完整度 {row['documentation_score']:.1f}、綠色效益分數 {row['green_impact_score']:.1f}、PD {row['pd_estimate']:.2%}
- 建議做法：先確認補件品質，再將專案效益、節能量、減碳量與還款來源對齊到授信敘述中。
""".strip(),
            "專案 CAPEX / 節能效益摘要": f"""### 專案 CAPEX / 節能效益摘要

- 專案：{row['transition_project']}
- 預估 CAPEX：{row['project_capex_twd_mn']:.1f} 百萬 TWD
- 預期節能：{row['expected_energy_saving_pct']:.0%}
- 預期減碳：{row['expected_emission_reduction_tons']:,.0f} 噸 CO2e
- 投資回收年期：{row['payback_years']:.1f} 年
- 建議說明重點：將專案用途、節能機制、量測方式與財務回收邏輯說清楚。
""".strip(),
            "授信條件與 KPI 建議草案": f"""### 授信條件與 KPI 建議草案

**建議條件**
- 資金用途需明確對應 {row['transition_project']} 專案。
- 撥款前應補齊主要缺件並確認報價或合約文件。
- 若為分階段撥款，可搭配里程碑驗證機制。

**建議貸後 KPI**
{_bullet_markdown(row['post_loan_kpis'])}
""".strip(),
            "貸後追蹤月報": f"""### 貸後追蹤月報（示意）

- 公司：{row['company_name']}
- 專案：{row['transition_project']}
- 本月追蹤重點：節能進度、設備導入、實際用電變化、再生能源占比與補件狀態。
- 建議追蹤 KPI：
{_bullet_markdown(row['post_loan_kpis'])}
""".strip(),
            "供應鏈議合建議書": f"""### 供應鏈議合建議書

- 供應鏈壓力來源：{row['supply_chain_role']}／{row['relationship_stage']}
- 主要議合方向：說明專案如何回應客戶減碳要求、揭露需求與交付穩定性。
- 建議議合重點：
{_bullet_markdown(row['bank_actions'])}
""".strip(),
            "訪廠訪談提綱": f"""### 訪廠訪談提綱

1. 請說明 {row['transition_project']} 的時程、責任部門與執行方式。
2. 請說明設備規格、報價依據與驗收標準。
3. 請說明專案完成後如何量測節能、減碳與產能影響。
4. 請說明還款來源與現金流安排。
5. 請說明客戶減碳要求與現有因應狀況。
""".strip(),
            "交叉銷售與合作夥伴推薦": f"""### 交叉銷售與合作夥伴推薦

- 主力金融產品：{row['recommended_product']}
- 可延伸合作：設備租賃、ESCO、碳盤查顧問、保險、綠電 / PPA、能源管理系統。
- 導入理由：有助於補強專案執行、數據揭露與貸後追蹤完整性。
""".strip(),
        }
        fallback = fallback_map.get(report_type, "尚未定義此報表類型。")

        if report_type in {"企業綠色授信申請摘要", "銀行前審 Memo"}:
            if report_type == "企業綠色授信申請摘要":
                return self.build_enterprise_brief(company_id)
            return self.build_bank_memo(company_id)

        docs = self._retrieve_docs([company_id], report_type, top_k=6)
        system_prompt = f"""
你是 Panda Copilot 的綠色授信報表生成助手。
請以繁體中文產出「{report_type}」，要求內容完整、詳細、專業，不要簡短帶過。
請只根據提供資料撰寫，不可虛構。
輸出格式請使用 Markdown，需有清楚段落、小標與可執行建議。
""".strip()
        user_prompt = f"請為下列公司生成「{report_type}」。\n\n結構化資料：\n{json.dumps(self._row_context(row), ensure_ascii=False, indent=2)}\n\n參考文件：\n{docs}"
        return self._llm_or_fallback(system_prompt, user_prompt, fallback)

    def answer_question(self, query: str) -> Dict[str, object]:
        route = self._route_query(query)
        if route["type"] == "single_company":
            return self._answer_single(query, route["company_ids"][0])
        if route["type"] == "company_comparison":
            return self._answer_compare(query, route["company_ids"])
        if route["type"] == "industry_benchmark":
            return self._answer_industry(query, route["company_ids"][0])
        return self._answer_generic(query)

    def _route_query(self, query: str) -> Dict[str, object]:
        company_ids = re.findall(r"CMP[_\- ]?(\d{3})", query, flags=re.I)
        company_ids = [f"CMP_{cid}" for cid in company_ids]
        if ("比較" in query or "compare" in query.lower()) and len(company_ids) >= 2:
            return {"type": "company_comparison", "company_ids": company_ids[:2]}
        if any(word in query for word in ["同業", "同產業", "產業平均", "benchmark"]) and company_ids:
            return {"type": "industry_benchmark", "company_ids": company_ids[:1]}
        if len(company_ids) == 1:
            return {"type": "single_company", "company_ids": company_ids}
        return {"type": "generic", "company_ids": company_ids}

    def _answer_single(self, query: str, company_id: str) -> Dict[str, object]:
        row = self.get_company(company_id)

        if any(word in query for word in ["缺件", "補件"]):
            answer = self.build_missing_doc_report(company_id)
            return {"answer": answer, "used_llm": False, "message": "使用平台規則式缺件報表。"}

        if "memo" in query.lower() or "前審" in query:
            text, used_llm, message = self.build_bank_memo(company_id)
            return {"answer": text, "used_llm": used_llm, "message": message}

        docs = self._retrieve_docs([company_id], query, top_k=6)
        fallback = self._enterprise_fallback(row)
        system_prompt = """
你是 Panda Copilot 的綠色授信專家顧問。
請針對單一企業問題提供完整、詳細、全面的繁體中文回答。
請先給出結論，再說明依據、風險、缺件、建議產品與下一步。
不可以簡短敷衍，不可以輸出 JSON、list 原始字串或 thinking 過程。
""".strip()
        user_prompt = f"問題：{query}\n\n結構化資料：\n{json.dumps(self._row_context(row), ensure_ascii=False, indent=2)}\n\n參考文件：\n{docs}"
        text, used_llm, message = self._llm_or_fallback(system_prompt, user_prompt, fallback)
        return {"answer": text, "used_llm": used_llm, "message": message}

    def _answer_compare(self, query: str, company_ids: List[str]) -> Dict[str, object]:
        left = self.get_company(company_ids[0])
        right = self.get_company(company_ids[1])
        better = left if left["green_loan_fit_score"] >= right["green_loan_fit_score"] else right
        fallback = f"""### 兩家公司比較結論

- 比較對象：{left['company_name']}（{left['company_id']}） vs {right['company_name']}（{right['company_id']}）
- 初步結論：{better['company_name']} 較適合優先推進綠色授信。

**主要原因**
- {left['company_name']}：Green Loan Fit {left['green_loan_fit_score']:.1f}、Panda Score {left['panda_score']:.1f}、建議產品 {left['recommended_product']}
- {right['company_name']}：Green Loan Fit {right['green_loan_fit_score']:.1f}、Panda Score {right['panda_score']:.1f}、建議產品 {right['recommended_product']}
- 優先推進對象：{better['company_name']}，因其整體適配度較高，且授信敘述較完整。
""".strip()
        docs = self._retrieve_docs(company_ids, query, top_k=10)
        payload = {left["company_id"]: self._row_context(left), right["company_id"]: self._row_context(right)}
        system_prompt = """
你是 Panda Copilot 的雙公司比較分析助手。
請用繁體中文提供完整、詳細、全面的比較結論。
至少要包含：
- 一、總結論
- 二、兩家公司優劣勢對照
- 三、授信可推進性差異
- 四、風險與缺件差異
- 五、建議先推進哪一家與原因
不要輸出 JSON 或 thinking 過程。
""".strip()
        user_prompt = f"問題：{query}\n\n結構化資料：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n參考文件：\n{docs}"
        text, used_llm, message = self._llm_or_fallback(system_prompt, user_prompt, fallback)
        return {"answer": text, "used_llm": used_llm, "message": message}

    def _answer_industry(self, query: str, company_id: str) -> Dict[str, object]:
        result = industry_comparison(self.scored, self.benchmarks, company_id)
        company = result["company"]
        mean = result["industry_mean"]
        fallback = (
            f"### 同產業比較\n\n"
            f"- 公司：{company['company_name']}（{company['company_id']}）\n"
            f"- 產業：{company['industry']}\n"
            f"- Panda Score：{company['panda_score']:.1f}，產業平均 {mean.get('panda_score', float('nan')):.1f}\n"
            f"- Green Loan Fit Score：{company['green_loan_fit_score']:.1f}，產業平均 {mean.get('green_loan_fit_score', float('nan')):.1f}\n"
            f"- 文件完整度：{company['documentation_score']:.1f}，產業平均 {mean.get('documentation_score', float('nan')):.1f}\n"
        )
        if not self._llm_ready():
            return {"answer": fallback, "used_llm": False, "message": "目前未啟用 LLM，已使用規則式產業比較。"}
        system_prompt = "你是 Panda Copilot 的產業 benchmark 分析助手。請用繁體中文提供完整產業比較解讀，說明該公司相對於同業的優勢、弱點、授信含義與建議。"
        user_prompt = f"問題：{query}\n\n公司資料：\n{json.dumps(company, ensure_ascii=False, indent=2)}\n\n產業平均：\n{json.dumps(mean, ensure_ascii=False, indent=2)}"
        text, used_llm, message = self._llm_or_fallback(system_prompt, user_prompt, fallback)
        return {"answer": text, "used_llm": used_llm, "message": message}

    def _answer_generic(self, query: str) -> Dict[str, object]:
        fallback = "請在問題中加入公司代碼（例如 CMP_001），或改用 Report Studio 生成指定報表。"
        if not self._llm_ready():
            return {"answer": fallback, "used_llm": False, "message": "目前未啟用 LLM。"}
        top_ids = self.scored.sort_values("green_loan_fit_score", ascending=False).head(3)["company_id"].tolist()
        docs = self._retrieve_docs(top_ids, query, top_k=8)
        system_prompt = """
你是 Panda Copilot 的總顧問。
請以繁體中文完整回答使用者問題，回覆要詳細、全面、專業，不要過度簡化。
若問題資料不足，請明確指出不足之處，並建議使用者加入公司代碼或指定報表。
不要輸出 JSON 或 thinking 過程。
""".strip()
        user_prompt = f"問題：{query}\n\n可參考文件：\n{docs}"
        text, used_llm, message = self._llm_or_fallback(system_prompt, user_prompt, fallback)
        return {"answer": text, "used_llm": used_llm, "message": message}
