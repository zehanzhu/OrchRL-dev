# Search MAS 运行说明书

本文档是当前仓库 Search MAS 训练链路的最短可执行 runbook，重点说明怎么启动，以及运行前必须配置哪些文件。

适用范围：

- 当前仓库内已经生效的 MATE 外接黑盒 MAS 训练路径
- 启动脚本：`scripts/run_search_mas_train_e2e.sh`
- 主配置：`orchrl/config/search/search_mas_nosearch_external*.yaml`

## 1. 最短启动命令

默认直接运行：

```bash
bash scripts/run_search_mas_train_e2e.sh
```

指定配置名运行：

```bash
CONFIG_NAME=search_mas_nosearch_external bash scripts/run_search_mas_train_e2e.sh
```

指定 GPU 运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 CONFIG_NAME=search_mas_nosearch_external_5step_4x4_conservative bash scripts/run_search_mas_train_e2e.sh
```

说明：

- 启动脚本会自动 source `scripts/utils/export_repo_pythonpath.sh`
- 会自动设置：

```bash
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
```

- 不需要你手动再 export 一次 `PYTHONPATH`

## 2. 运行前最重要的一件事

当前训练并不是只靠 OrchRL 仓库内部就能跑起来。

它依赖三部分同时对齐：

1. OrchRL 仓库内的训练配置
2. 仓外的 MATE 配置模板文件
3. 仓外的 Search MAS 外部应用目录、数据文件和模型目录

所以你真正要检查的不是“一个 YAML”，而是一组互相引用的路径。

## 3. 必须配置的文件

下面这张表是当前链路里最关键的文件清单。

| 文件 | 作用 | 是否通常需要修改 | 不对齐会怎样 |
| --- | --- | --- | --- |
| `scripts/run_search_mas_train_e2e.sh` | 训练入口脚本，决定默认配置、GPU、日志路径 | 有时需要 | 启动不到你想要的配置或 GPU |
| `orchrl/config/search/search_mas_nosearch_external.yaml` | 主训练配置，定义模型路径、prompt 路径、外部 MAS 路径、reward provider | 必须按环境检查 | 启动前校验直接失败，或训练接错外部系统 |
| `orchrl/config/search/search_mas_nosearch_external_5step_4x4_conservative.yaml` | 轻量实跑配置，覆盖 step/batch 等训练参数 | 视运行目标而定 | 可能跑的是默认 smoke 配置，而不是你想要的训练规模 |
| `/data1/zzh/MATE-main/artifacts/search_mas_real_nosearch.yaml` | 外部 MATE 模板配置，控制黑盒 MAS 应用行为 | 必须按外部系统检查 | MAS 能启动但行为不符合预期，或外部依赖不可用 |
| `/data1/zzh/mas_app/search/scripts/run_search_mas.py` | 外部黑盒 MAS 启动脚本 | 通常不改，但必须存在 | MATE 无法拉起外部 MAS |
| `orchrl/reward/search/external_mas_reward.py` | 当前 reward 实现 | 通常不改，但导入路径必须对 | rollout 完成后 reward 计算失败 |

## 4. 启动脚本里要看什么

文件：

- [run_search_mas_train_e2e.sh](/data1/zzh/ZZH_0209_V3/OrchRL/scripts/run_search_mas_train_e2e.sh)

这个脚本当前会控制：

- 默认配置名：`search_mas_nosearch_external_5step_4x4_conservative`
- 默认 GPU：`3,4,5`
- 默认日志路径：`logs/search_mas_train_e2e_<timestamp>.log`

你通常需要检查这几个点：

- `DEFAULT_CONFIG_NAME` 是否是你想跑的配置
- `DEFAULT_CUDA_VISIBLE_DEVICES` 是否和当前机器 GPU 分配一致
- 是否要通过环境变量覆盖 `CONFIG_NAME` 或 `CUDA_VISIBLE_DEVICES`

脚本还会在真正启动前，从 Hydra 配置里解析出并检查这些路径是否存在：

- `training.mate.mas_work_dir`
- `training.mate.config_template_path`
- `training.mate.prompt_loader.path`
- `base_models.policy_0.path`
- `base_models.policy_1.path`
- `base_models.policy_2.path`

只要其中一个不存在，脚本会直接报错退出。

## 5. 主训练配置里必须检查的字段

文件：

- [search_mas_nosearch_external.yaml](/data1/zzh/ZZH_0209_V3/OrchRL/orchrl/config/search/search_mas_nosearch_external.yaml)

这是当前最核心的配置文件。下面这些字段是运行前必须检查的。

### 5.1 模型路径

必须检查：

- `base_models.policy_0.path`
- `base_models.policy_1.path`
- `base_models.policy_2.path`

当前默认值都指向：

- `/data1/lll/models/Qwen3-0.6B`

如果你的模型目录不同，这里必须改。

### 5.2 外部 MAS 工作目录

必须检查：

- `training.mate.mas_work_dir`

当前值：

- `/data1/zzh/mas_app/search`

这个目录必须真实存在，因为外部 Search MAS 就在这里运行。

### 5.3 外部 MAS 启动命令

必须检查：

- `training.mate.mas_command_template`

当前值：

```bash
python /data1/zzh/mas_app/search/scripts/run_search_mas.py --config {config_path} --question {prompt}
```

如果你把外部 MAS 仓库搬了位置，或者启动脚本换了名字，这里必须同步修改。

### 5.4 MATE 外部模板文件

必须检查：

- `training.mate.config_template_path`

当前值：

- `/data1/zzh/MATE-main/artifacts/search_mas_real_nosearch.yaml`

这个文件不是 OrchRL 仓库内文件，但它是当前链路的必要配置文件，不能漏。

### 5.5 Prompt 数据集路径

必须检查：

- `training.mate.prompt_loader.path`

当前值：

- `/data1/zzh/mas_app/search/data/drmas_search_mas/test_sampled.parquet`

如果数据集路径变了，这里必须改，否则启动前校验会直接失败。

### 5.6 Reward Provider

必须检查：

- `training.mate.reward.provider`

当前值：

- `orchrl.reward.search.external_mas_reward:compute_reward`

如果你改了 reward 文件位置或函数名，这里必须同步改，否则会在 reward provider 导入阶段失败。

## 6. 5-step 配置文件要看什么

文件：

- [search_mas_nosearch_external_5step_4x4_conservative.yaml](/data1/zzh/ZZH_0209_V3/OrchRL/orchrl/config/search/search_mas_nosearch_external_5step_4x4_conservative.yaml)

这个文件继承主配置，只覆盖更偏 smoke / e2e 验证的一组训练参数。

你主要检查：

- `training.total_training_steps`
- `training.train_batch_size`
- `training.train_sample_num`
- `training.val_freq`
- `training.model_checkpoints_dir`
- `training.experiment_name`

默认启动脚本跑的就是它。

如果你想先做短链路验证，保留它。
如果你想跑正式一点的配置，切回 `search_mas_nosearch_external`。

## 7. 仓外 MATE 模板文件里必须检查什么

文件：

- `/data1/zzh/MATE-main/artifacts/search_mas_real_nosearch.yaml`

这个文件虽然不在当前仓库里，但它是当前训练链路里必须对齐的配置文件。

建议你至少检查这些字段：

- `application.max_turns`
- `application.force_final_answer_on_max_turn`
- `search.provider`
- `search.retrieval_service_url`
- `agents.verifier.*`
- `agents.searcher.*`
- `agents.answerer.*`
- `output.prediction_path`
- `output.report_path`

当前要特别注意两点：

1. `search.provider` 现在是 `disabled`
2. `search.retrieval_service_url` 指向 `http://127.0.0.1:18080/retrieve`

如果你切回真实检索链路，就不能只改 OrchRL 配置，还必须把这个模板里的检索相关字段一起改对。

补充说明：

- 模板里的 `llm.base_url`、`llm.model` 是外部应用默认值
- 当前训练链路下，真正的 role 到 backend 映射是 OrchRL 运行时通过 MATE 注入的
- 所以这两个字段可以保留模板默认值，但 role 映射链路必须整体通

## 8. 哪些文件通常不用改，但必须存在

这些文件一般不是“配置入口”，但缺了就跑不起来：

- `/data1/zzh/mas_app/search/scripts/run_search_mas.py`
- `orchrl/reward/search/external_mas_reward.py`
- `scripts/utils/export_repo_pythonpath.sh`

含义分别是：

- 外部 MAS 启动器
- 训练 reward 实现
- 仓库内 `PYTHONPATH` 注入脚本

## 9. 启动前检查清单

正式运行前，至少确认下面这些项：

1. 当前机器上 `CUDA_VISIBLE_DEVICES` 指向的 GPU 真能用
2. `scripts/run_search_mas_train_e2e.sh` 里的默认配置是你要跑的那个
3. `search_mas_nosearch_external.yaml` 里的 3 个模型路径都存在
4. `training.mate.mas_work_dir` 指向真实的外部 Search MAS 目录
5. `training.mate.mas_command_template` 能正确找到 `run_search_mas.py`
6. `training.mate.config_template_path` 指向真实存在的 MATE 模板文件
7. `training.mate.prompt_loader.path` 指向真实存在的数据文件
8. `training.mate.reward.provider` 和实际 reward 文件位置一致
9. 如果外部模板启用了检索，检索服务地址也必须可达

## 10. 推荐运行方式

### 10.1 先做 5-step 联通性验证

```bash
bash scripts/run_search_mas_train_e2e.sh
```

适合：

- 确认整条链路能不能跑通
- 检查外部 MAS、vLLM、reward 和 PPO 更新有没有接上

### 10.2 切回主配置运行

```bash
CONFIG_NAME=search_mas_nosearch_external bash scripts/run_search_mas_train_e2e.sh
```

适合：

- 使用主配置里的训练步数和默认 batch 参数

### 10.3 显式指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2 CONFIG_NAME=search_mas_nosearch_external bash scripts/run_search_mas_train_e2e.sh
```

适合：

- 多人共享机器
- 默认 GPU 号不适合当前环境

## 11. 日志怎么看

默认日志会输出到：

- `logs/search_mas_train_e2e_<timestamp>.log`

启动后可以重点看：

- 是否成功进入 Ray 初始化
- 是否成功启动每个 policy 的 rollout server
- 是否成功收集 MATE episodes
- 是否进入 PPO update
- `mean_reward`、`advantages`、`returns` 是否仍然全零

## 12. 常见报错先查哪

### 12.1 `Config file not found`

先查：

- `CONFIG_NAME`
- `orchrl/config/search/` 下是否有对应 YAML

### 12.2 `Required path not found`

先查主配置里的这些字段：

- `training.mate.config_template_path`
- `training.mate.prompt_loader.path`
- `base_models.policy_*.path`

### 12.3 外部 MAS 没启动起来

先查：

- `training.mate.mas_work_dir`
- `training.mate.mas_command_template`
- `/data1/zzh/mas_app/search/scripts/run_search_mas.py` 是否存在

### 12.4 Reward provider 导入失败

先查：

- `training.mate.reward.provider`
- [external_mas_reward.py](/data1/zzh/ZZH_0209_V3/OrchRL/orchrl/reward/search/external_mas_reward.py)

## 13. 最小结论

如果你只想记住最关键的几项，那就是先检查这 4 个路径：

1. `base_models.policy_{0,1,2}.path`
2. `training.mate.mas_command_template`
3. `training.mate.config_template_path`
4. `training.mate.prompt_loader.path`

这四类路径一旦不对，当前 Search MAS 训练链路基本就起不来。
