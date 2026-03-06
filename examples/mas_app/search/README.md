# Search MAS (Inference-Only)

本目录是从 DrMAS `search` 强化学习用例抽取出的推理版 MAS 应用，仅保留：
- 多 Agent 编排（Verifier 路由 + Search/Answer 执行）
- LLM 推理调用（OpenAI 兼容接口）
- 数据准备与黑盒验证

不包含训练逻辑与训练依赖。

## 目录结构

```text
MAS_APP/search
├── configs/
│   └── search_mas_example.yaml
├── requirements.txt
├── scripts/
│   ├── prepare_drmas_search_data.py
│   ├── deploy_searchr1_retrieval_service.sh
│   ├── run_opensearch_retrieval_service.py
│   ├── run_search_mas.py
│   └── validate_search_mas.py
└── search_mas/
    ├── core/              # 通用配置/LLM/Agent基类
    └── apps/search/       # Search MAS 业务实现
```

## 任务流程

1. Verifier Agent 先判断历史信息是否足够：`<verify>yes|no</verify>`  
2. 若 `no`：Search Agent 生成 `<search>query</search>`，调用检索后将结果写入 `<information>...</information>` 上下文  
3. 若 `yes`：Answer Agent 生成 `<answer>final answer</answer>`  
4. 达到最大轮数仍未路由到 answer 时，可按配置强制执行一次 Answer Agent（`force_final_answer_on_max_turn=true`）

## 快速开始

### 1) 安装依赖

```bash
cd /data1/lll/workspace/multi_agent_rl/DrMAS/MAS_APP/search
python -m pip install -r requirements.txt
```

### 2) 准备数据（与原 RL 用例一致的数据源）

```bash
python scripts/prepare_drmas_search_data.py \
  --hf_repo_id PeterJinGo/nq_hotpotqa_train \
  --output_dir ~/data/drmas_search_mas \
  --samples_per_source 30 \
  --save_jsonl
```

若你已提前下载数据到本地（例如放在 `$HF_HOME` 下），可离线准备：

```bash
export HF_HOME=/path/to/local/hf_home
python scripts/prepare_drmas_search_data.py \
  --hf_repo_id PeterJinGo/nq_hotpotqa_train \
  --output_dir ~/data/drmas_search_mas \
  --samples_per_source 30 \
  --save_jsonl \
  --local_only
```

说明：
- 脚本会优先在 `--dataset_root`（未传则自动使用 `$HF_HOME`）查找本地 `train.parquet` 和 `test.parquet`
- 若本地目录结构不是标准布局，也可直接传 `--train_file /path/train.parquet --test_file /path/test.parquet`
- 若需要远程下载兜底，去掉 `--local_only`（可选加 `--use_mirror`）

输出包括：
- `train.parquet`
- `test.parquet`
- `test_sampled.parquet`

每条数据包含：
- `question`
- `expected_answer`
- `expected_answers`

### 3) 启动 OpenSearch 检索服务（FastAPI `/retrieve`）

```bash
python scripts/run_opensearch_retrieval_service.py \
  --host 0.0.0.0 \
  --port 18080 \
  --opensearch-url http://127.0.0.1:9200 \
  --index drmas_search_docs \
  --fields contents,content,text \
  --auto-create-index
```

说明：
- 检索接口地址即 `http://127.0.0.1:18080/retrieve`
- 请求体兼容本项目 SearchClient：`{"query":"...", "topk":3, "return_scores":true}`
- 你的 OpenSearch 文档至少需要有 `contents/content/text` 其中一个字段

将检索 URL 写入环境变量（推荐）：

```bash
export SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:18080/retrieve
```

可用以下命令快速验证：

```bash
curl -X POST http://127.0.0.1:18080/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"Who won the 2022 FIFA World Cup?","topk":3,"return_scores":true}'
```

### 3.1) 一键部署 DrMAS 原生 SearchR1 本地检索服务（可选）

如果你希望复用 `DrMAS/README.md` 第 112-129 行的检索服务部署方式（下载 `wiki-18` 索引与语料并启动本地检索服务），可直接运行：

```bash
bash scripts/deploy_searchr1_retrieval_service.sh
```

说明：
- 该脚本封装了：创建 `retriever` conda 环境并安装依赖（`numpy/torch/faiss-gpu/uvicorn/fastapi` 等）、下载索引与语料、合并 `part_*` 为 `e5_Flat.index`、解压 `wiki-18.jsonl.gz`、启动 `/retrieve` 服务。
- 默认使用 `conda` 环境 `retriever`、数据目录 `/data1/lll/datasets/wiki-18`、端口 `8010`、模型 `/data1/lll/models/e5-base-v2`，日志写入 `retrieval_server.log`。
- 推荐通过环境变量配置：

```bash
export SEARCH_MAS_SEARCHR1_LOCAL_DIR=/data1/lll/datasets/wiki-18
export SEARCH_MAS_SEARCHR1_PORT=8010
export SEARCH_MAS_SEARCHR1_RETRIEVER_MODEL=/data1/lll/models/e5-base-v2
export SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:${SEARCH_MAS_SEARCHR1_PORT}/retrieve
```

- 也可通过命令参数覆盖（命令参数优先级高于环境变量）：

```bash
bash scripts/deploy_searchr1_retrieval_service.sh \
  --local-dir /data1/lll/datasets/wiki-18 \
  --conda-env retriever \
  --port 8010 \
  --retriever-model /data1/lll/models/e5-base-v2 \
  --log-file retrieval_server.log
```

- 若你已手动准备好环境，可加 `--skip-env-setup` 跳过 conda 环境创建与依赖安装。
- 若环境已存在但希望重装依赖，可加 `--force-env-setup`。
- 若使用该服务，建议设置 `export SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve`（或使用你自定义的端口）。
- 与上面的 OpenSearch 检索服务是替代关系，二选一即可。

### 4) 配置环境变量（OpenAI / vLLM）

本项目支持用环境变量覆盖 YAML 中的 LLM 配置，推荐用环境变量切换 OpenAI 官方接口和本地 vLLM。

全局 LLM 变量（优先级高于 YAML）：
- `SEARCH_MAS_LLM_BASE_URL`
- `SEARCH_MAS_LLM_API_KEY`
- `SEARCH_MAS_LLM_MODEL`

兼容别名（等价可选）：
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

可选扩展：
- 按 Agent 单独覆盖：`SEARCH_MAS_VERIFIER_LLM_*`、`SEARCH_MAS_SEARCHER_LLM_*`、`SEARCH_MAS_ANSWERER_LLM_*`
- 检索地址：`SEARCH_MAS_RETRIEVAL_SERVICE_URL`

OpenAI 官方接口示例：

```bash
export SEARCH_MAS_LLM_BASE_URL=https://api.openai.com/v1
export SEARCH_MAS_LLM_API_KEY=sk-xxx
export SEARCH_MAS_LLM_MODEL=gpt-4.1-mini
```

vLLM 本地接口示例：

```bash
export SEARCH_MAS_LLM_BASE_URL=http://127.0.0.1:8000/v1
export SEARCH_MAS_LLM_API_KEY=EMPTY
export SEARCH_MAS_LLM_MODEL=/data1/lll/models/Qwen3-4B-Instruct-2507
```

### 5) 单题推理

```bash
python scripts/run_search_mas.py \
  --config configs/search_mas_example.yaml \
  --question "Who won the 2014 FIFA World Cup?"
```

### 6) 黑盒验证（全量遍历 + 打分 + 准确率）

```bash
python scripts/validate_search_mas.py \
  --config configs/search_mas_example.yaml \
  --input_file ~/data/drmas_search_mas/test_sampled.parquet
```

输出：
- `output.prediction_path`：逐样本预测与差异
- `output.report_path`：总体正确率与分源统计

## 配置说明

`configs/search_mas_example.yaml` 关键字段：
- `llm.base_url` + `llm.api_key` + `llm.model`：OpenAI 兼容调用（OpenAI / vLLM 都可）
  - 推荐通过环境变量覆盖：`SEARCH_MAS_LLM_BASE_URL` / `SEARCH_MAS_LLM_API_KEY` / `SEARCH_MAS_LLM_MODEL`
  - 兼容别名：`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`
- `search.retrieval_service_url`：检索服务地址（兼容 DrMAS Search-R1 本地检索服务）
  - 推荐通过环境变量覆盖：`SEARCH_MAS_RETRIEVAL_SERVICE_URL`
- `application.max_turns`：最大多轮步数
- `agents.<agent>.llm.*`：可为 `verifier/searcher/answerer` 单独覆盖 LLM 后端（`base_url/api_key/timeout/max_retries/retry_backoff_sec/model`）
  - 环境变量前缀：`SEARCH_MAS_VERIFIER_LLM_*` / `SEARCH_MAS_SEARCHER_LLM_*` / `SEARCH_MAS_ANSWERER_LLM_*`
- `agents.*`：不同 Agent 的采样参数（`temperature/top_p/max_tokens/stop/extra_body`）
- `validation.use_substring_em`：验证模式（EM 或 SubEM）

示例（vLLM + OpenSearch 检索服务分离部署）：

```yaml
llm:
  base_url: http://127.0.0.1:8000/v1

agents:
  verifier:
    llm:
      base_url: http://127.0.0.1:8001/v1
      model: /data1/lll/models/Qwen3-4B-Instruct-2507
  searcher:
    llm:
      base_url: http://127.0.0.1:8002/v1
  answerer:
    # 不配置 llm 时会继承顶层 llm 配置
    temperature: 0.4

search:
  provider: http
  retrieval_service_url: ${SEARCH_MAS_RETRIEVAL_SERVICE_URL}
```
