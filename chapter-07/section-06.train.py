"""
从头训练因果语言模型 (GPT-2 on CodeParrot-DS)
章节：HuggingFace LLM Course Chapter 7 Section 6

目标：在 codeparrot-ds 数据集上从头训练一个 GPT-2 模型，
      使其能够补全 Python 数据科学相关代码（pandas/sklearn/matplotlib）。

硬件配置：2x H100 NVL 94GB
启动命令：torchrun --nproc_per_node=2 section-06.train.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# 依赖安装（首次运行前执行）：
#   pip install datasets transformers[sentencepiece] accelerate
# ──────────────────────────────────────────────────────────────────────────────

from collections import defaultdict

import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

# ──────────────────────────────────────────────────────────────────────────────
# 登录 HuggingFace Hub（用于保存和推送模型）
# ──────────────────────────────────────────────────────────────────────────────
login(token=os.environ["HF_TOKEN"])  # 也可通过环境变量 HF_TOKEN 传入，避免交互式登录：login(token=os.environ["HF_TOKEN"])

# ──────────────────────────────────────────────────────────────────────────────
# 1. 加载数据集
#
# codeparrot-ds 是从 GitHub 爬取的 Python 代码，经过过滤，
# 只保留了包含 pandas/sklearn/matplotlib/seaborn 关键词的文件。
# 训练集约 60 万条，验证集约 3000 条。
# ──────────────────────────────────────────────────────────────────────────────
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="train")

raw_datasets = DatasetDict(
    {
        "train": ds_train,
        "valid": ds_valid,
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. 加载 Tokenizer
#
# 使用在 CodeSearchNet 数据集上训练的 BPE tokenizer，
# 词表专门针对代码优化，比通用 GPT-2 tokenizer 更适合 Python 代码。
# context_length=128 表示每个训练样本的 token 长度。
# ──────────────────────────────────────────────────────────────────────────────
context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Tokenize 数据集
#
# return_overflowing_tokens=True：将长文本切分为多个 context_length 大小的块，
#   而不是直接截断，避免浪费训练数据。
# return_length=True：返回每个块的实际长度，用于过滤不足 context_length 的块。
# 只保留长度恰好等于 context_length 的块，确保所有样本等长，
#   这样 DataCollator 无需 padding，训练效率更高。
#
# H100 NVL 优化：num_proc=64 使用 64 个 CPU 进程并行处理，
#   tokenizers 库内部有 Rust 多线程，num_proc>1 时会自动禁用以避免死锁。
# ──────────────────────────────────────────────────────────────────────────────
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,  # 长文本切块而非截断
        return_length=True,              # 返回每块长度
    )
    # 只保留长度恰好为 context_length 的完整块，丢弃尾部不完整的块
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,                                        # 批量处理，速度更快
    remove_columns=raw_datasets["train"].column_names,  # 删除原始列，只保留 input_ids
    batch_size=2000,                                     # 每批处理 2000 条原始样本
    num_proc=64,                                         # 64 进程并行（128 核机器留余量给 OS）
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. 初始化模型
#
# 从头初始化 GPT-2 配置，而非加载预训练权重。
# 关键参数：
#   vocab_size：使用自定义 tokenizer 的词表大小（约 32K）
#   n_ctx：上下文长度，与 context_length 保持一致
#   bos/eos_token_id：句子起止符，用于生成时的边界控制
#
# ──────────────────────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# GPT-2 模型较小且 context_length=128，Flash Attention 收益可忽略，使用标准注意力
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Data Collator
#
# DataCollatorForLanguageModeling 负责将样本拼成 batch，并自动生成 labels。
# mlm=False：使用因果语言模型（CLM）目标，即预测下一个 token，
#   而非 BERT 的掩码语言模型（MLM）。
# labels 与 input_ids 完全相同，loss 在模型内部通过 shift 计算（预测下一位）。
# pad_token 设为 eos_token：GPT-2 原本没有 pad token，复用 eos 即可。
# ──────────────────────────────────────────────────────────────────────────────
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ──────────────────────────────────────────────────────────────────────────────
# 6. 训练参数
#
# 【2x H100 NVL 关键配置说明】
#
# per_device_train_batch_size=128：
#   H100 NVL 有 94GB 显存，GPT-2 (124M) 很小，每卡 batch=128 绰绰有余。
#   原始单卡 4090 设置为 32，H100 可以提高 4x。
#
# gradient_accumulation_steps=2：
#   有效 batch size = 128(每卡) × 2(卡数) × 2(累积) = 512。
#   原始配置有效 batch = 32 × 1 × 8 = 256，适当增大有助于训练稳定性。
#
# bf16=True（替代原来的 fp16=True）：
#   H100 原生支持 BF16，比 FP16 数值范围更大（不易溢出），
#   训练稳定性更好，且速度相同。4090 也支持 bf16，推荐使用。
#
# torch_compile=True：
#   使用 torch.compile 对模型进行编译优化（PyTorch 2.0+），
#   首次运行会有几分钟编译开销，之后每步可提速 20-30%。
#
# optim="adamw_torch_fused"：
#   使用 CUDA Fused AdamW，比标准 AdamW 更快，减少显存碎片。
#
# dataloader_num_workers=8：
#   用 8 个子进程预加载数据，避免 GPU 等待 CPU IO。
#
# ddp_find_unused_parameters=False：
#   多卡 DDP 训练时，明确告知所有参数都参与梯度计算，
#   避免 DDP 做不必要的检查，提升多卡效率。
# ──────────────────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="codeparrot-ds",

    # ── Batch 配置 ──────────────────────────────────────
    per_device_train_batch_size=128,     # 每卡 batch（H100 NVL 94GB 可轻松承载）
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=2,       # 有效 batch = 128 × 2卡 × 2 = 512

    # ── 精度配置 ─────────────────────────────────────────
    bf16=True,                           # H100 推荐使用 bf16（比 fp16 更稳定）

    # ── 性能优化 ─────────────────────────────────────────
    torch_compile=True,                  # torch.compile 编译加速（PyTorch 2.0+）
    optim="adamw_torch_fused",           # Fused AdamW，更快更省显存
    dataloader_num_workers=8,            # 数据预加载进程数
    dataloader_pin_memory=True,          # 锁页内存，加速 CPU→GPU 传输

    # ── 多卡配置 ─────────────────────────────────────────
    ddp_find_unused_parameters=False,    # 所有参数均参与训练，关闭无用参数检查

    # ── 训练超参 ─────────────────────────────────────────
    num_train_epochs=1,
    weight_decay=0.1,                    # L2 正则化，防止过拟合
    warmup_steps=1_000,                  # 学习率从 0 线性增加到 learning_rate 的步数
    lr_scheduler_type="cosine",          # 余弦退火调度，训练后期平滑降低学习率
    learning_rate=5e-4,

    # ── 评估与保存 ────────────────────────────────────────
    eval_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    save_steps=5_000,
    save_total_limit=2,                  # 最多保留 2 个 checkpoint（最近 1 个 + 最优 1 个）
    load_best_model_at_end=True,         # 训练结束后自动加载最优 checkpoint
    metric_for_best_model="eval_loss",   # 以验证集 loss 为准判断最优（loss 越低越好）
    greater_is_better=False,             # loss 越小越好

    # ── Hub 推送 ─────────────────────────────────────────
    push_to_hub=True,                    # 每次 save 自动推送到 HuggingFace Hub
)

# ──────────────────────────────────────────────────────────────────────────────
# 7. 初始化 Trainer
# ──────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

# ──────────────────────────────────────────────────────────────────────────────
# 8. 开始训练
#
# 启动命令（2x H100 NVL）：
#   torchrun --nproc_per_node=2 section-06.train.py
#
# Trainer 会自动：
#   - 通过 DDP（DistributedDataParallel）将模型复制到每张卡
#   - 每张卡处理不同的 mini-batch，梯度 all-reduce 后同步更新
#   - 只在主进程（rank 0）做 eval、save 和 push_to_hub
# ──────────────────────────────────────────────────────────────────────────────
trainer.train()
