# 双模型支持使用说明

## 功能概述

本项目已支持在 CodeSIM 算法中使用两个不同的模型，每次调用模型时可以灵活选择使用哪个模型。

## 配置步骤

### 1. 环境变量配置

在 `.env` 文件中配置两个模型的 API 端点：

```bash
# 设置 API 类型为 openai
API_TYPE="openai"

# API Key（本地 vLLM 通常设为 EMPTY）
OPENAI_API_KEY=EMPTY

# Model 1 的 API 端点
OPENAI_API_BASE=http://localhost:8000/v1

# Model 2 的 API 端点（可选）
OPENAI_API_BASE_2=http://localhost:8001/v1
```

### 2. 启动 vLLM 服务

在两个不同的端口启动两个模型：

```bash
# Terminal 1: 启动 Model 1 (例如 0.5B)
vllm serve /path/to/Qwen2.5-Coder-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4

# Terminal 2: 启动 Model 2 (例如 7B)
vllm serve /path/to/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4
```

## 使用方式

### 单模型模式（向后兼容）

```bash
# 方式1：使用原有的 --model 参数
python src/main.py \
  --dataset HumanEval \
  --strategy CodeSIM \
  --model_provider openai \
  --model "/path/to/model" \
  --temperature 0 \
  --top_p 0.95 \
  --pass_at_k 1 \
  --language Python3

# 方式2：使用新的 --model1 参数
python src/main.py \
  --dataset HumanEval \
  --strategy CodeSIM \
  --model_provider openai \
  --model1 "/path/to/model" \
  --temperature 0 \
  --top_p 0.95 \
  --pass_at_k 1 \
  --language Python3
```

### 双模型模式

```bash
python src/main.py \
  --dataset HumanEval \
  --strategy CodeSIM \
  --model_provider openai \
  --model1 "/path/to/Qwen2.5-Coder-0.5B-Instruct" \
  --model2 "/path/to/Qwen2.5-Coder-7B-Instruct" \
  --temperature 0 \
  --top_p 0.95 \
  --pass_at_k 1 \
  --language Python3 \
  --cont yes \
  --result_log partial \
  --verbose 2 \
  --store_log_in_file yes
```

## 代码中指定使用的模型

在 `src/promptings/CodeSIM.py` 中，所有 `gpt_chat` 调用默认使用 `use_model=1`。
你可以根据需要修改不同阶段使用的模型：

```python
# Planning 阶段使用 Model 1
response = self.gpt_chat(
    processed_input=input_for_planning,
    use_model=1  # 修改为 2 则使用 Model 2
)

# Code Generation 阶段
response = self.gpt_chat(input_for_final_code_generation, use_model=1)

# Debugging 阶段
response = self.gpt_chat(input_for_debugging, use_model=1)
```

## 输出说明

### 日志文件（Log.txt）

在日志开头会显示模型配置：

```
======================================================================
=== Model Configuration ===
Model 1: /path/to/Qwen2.5-Coder-0.5B-Instruct
Model 2: /path/to/Qwen2.5-Coder-7B-Instruct
======================================================================
```

每次模型响应时会标注使用的模型：

```
______________________________________________________________________
Response from final code generation (Model 1):

<response content>
```

### 结果路径

- **单模型模式**：`results/HumanEval/CodeSIM/{model1_name}/...`
- **双模型模式**：`results/HumanEval/CodeSIM/{model1_name}+{model2_name}/...`

例如：`results/HumanEval/CodeSIM/Qwen2.5-Coder-0.5B+Qwen2.5-Coder-7B/Python3-0.0-0.95-1/Run-1/`

### Summary-ET.txt

双模型模式下，Summary 文件会包含模型配置和统计信息：

```
======================================================================
=== Model Configuration ===
Model 1: /path/to/Qwen2.5-Coder-0.5B-Instruct
Model 2: /path/to/Qwen2.5-Coder-7B-Instruct
======================================================================

Accuracy:                            81.7
Solved:                               134
...

======================================================================
=== Model-wise Statistics ===

Model 1 Statistics:
  Total Api Calls:                 320
  Total Prompt Tokens:          180000
  Total Completion Tokens:      170000

Model 2 Statistics:
  Total Api Calls:                 428
  Total Prompt Tokens:          252361
  Total Completion Tokens:      247401
```

## 错误处理

如果代码中调用了未配置的模型，会抛出异常：

```python
ValueError: Model 2 is not configured but was requested
```

## 实现细节

### 修改的文件

1. **`src/main.py`**
   - 添加 `--model1` 和 `--model2` 命令行参数
   - 支持向后兼容（保留 `--model` 参数）
   - 初始化两个模型实例
   - 传递模型统计信息给 summary 生成函数

2. **`src/models/OpenAI.py`**
   - 支持通过 `api_base_env_var` 参数指定不同的环境变量

3. **`src/promptings/Base.py`**
   - `__init__` 方法添加 `model2`, `model1_path`, `model2_path` 参数
   - 添加 `model1_stats` 和 `model2_stats` 统计字典
   - `gpt_chat` 方法添加 `use_model` 参数（默认为 1）
   - 自动统计每个模型的调用次数和 token 使用量

4. **`src/promptings/CodeSIM.py`**
   - 在 `run_single_pass` 开头打印模型配置
   - 所有 `gpt_chat` 调用添加 `use_model=1` 参数
   - 日志输出中显示使用的模型

5. **`src/utils/summary.py`**
   - `gen_summary` 函数添加 `model_stats` 可选参数
   - 在 Summary 开头写入模型配置
   - 在 Summary 末尾写入模型统计信息

6. **`.env.example`**
   - 添加 `OPENAI_API_BASE` 和 `OPENAI_API_BASE_2` 配置示例

## 注意事项

1. 两个模型共用相同的 `temperature`、`top_p` 等参数
2. 默认情况下，所有阶段都使用 Model 1
3. 需要手动修改 `CodeSIM.py` 中的 `use_model` 参数来改变模型选择策略
4. 确保两个 vLLM 服务在运行前已正确启动
5. 模型统计只在双模型模式下显示（model2 不为 None）

## 后续扩展

如需实现自动模型选择逻辑，可以在 `CodeSIM.py` 中添加策略函数：

```python
def select_model_for_stage(stage: str) -> int:
    """
    根据阶段选择模型
    stage: 'planning', 'simulation', 'code_generation', 'debugging'
    返回: 1 或 2
    """
    if stage in ['planning', 'simulation']:
        return 1  # 使用小模型
    else:
        return 2  # 使用大模型
```

然后在调用处：

```python
response = self.gpt_chat(
    processed_input=input_for_planning,
    use_model=select_model_for_stage('planning')
)
```

