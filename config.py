from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    app_title: str = "Panda Copilot"
    app_subtitle: str = "第三方綠色授信協作平台"

    # Ollama / Qwen 設定
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen3.6:27b"
    llm_timeout_sec: int = 240
    llm_keep_alive: str = "60m"
    llm_num_ctx: int = 4096
    llm_num_predict: int = 1200
    llm_temperature: float = 0.20

    # 資料模擬 / 計算設定
    random_seed: int = 42
    n_companies: int = 120
    monthly_periods: int = 24
    start_date: str = "2024-01-01"
    taiwan_grid_factor_kg_per_kwh: float = 0.474
    default_carbon_price_twd_per_ton: float = 300.0

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data"
        self.docs_dir = self.root_dir / "company_docs"
        self.reports_dir = self.root_dir / "reports"
        for path in (self.data_dir, self.docs_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def monthly_dates(self):
        import pandas as pd

        return pd.date_range(self.start_date, periods=self.monthly_periods, freq="MS")


CONFIG = AppConfig()
