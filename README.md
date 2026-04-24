# Panda Copilot

Panda Copilot 是一個面向 **銀行端與企業端** 的第三方綠色授信協作平台，支援：

- 企業送件資料整理
- 銀行前審 Memo 生成
- 缺件與補件清單
- 雙公司比較與同業 Benchmark
- 專案 CAPEX / 節能 / 減碳效益摘要
- Copilot Console 問答

本版已針對本地 **Ollama + qwen3.6:27b** 做完整優化：

- 預設模型改為 `qwen3.6:27b`
- API 呼叫已加入 `think: false`
- 回覆風格預設為 **完整、詳細、全面**
- 若 Ollama 或模型不可用，會自動退回規則式輸出
- 側邊欄會顯示模型狀態與目前常駐模型
- 修正清單欄位被逐字顯示的問題

## 1. 安裝套件

```bash
pip install -r requirements.txt
```

## 2. 安裝模型

```bash
ollama pull qwen3.6:27b
```

## 3. 啟動平台

```bash
streamlit run app.py
```

## 4. 常見問題

### 已出現 `bind: Only one usage of each socket address...`

代表 Ollama 背景服務通常已經啟動，不需要再執行 `ollama serve`。

### 想確認模型可用

```bash
ollama run qwen3.6:27b "請用一句話介紹你自己"
```

### 想確認目前常駐模型

```bash
ollama ps
```

### 想查看已安裝模型

```bash
ollama ls
```
