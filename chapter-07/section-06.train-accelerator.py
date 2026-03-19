"""
从头训练因果语言模型 (GPT-2 on CodeParrot-DS) — Accelerate 自定义训练循环版
章节：HuggingFace LLM Course Chapter 7 Section 6

与 section-06.train.py（Trainer 版）的核心区别：
  1. 自定义加权 Loss：对含 plt/pd/sklearn/fit/predict 等关键 token 的样本加权，
     使模型更专注于学习数据科学相关的代码模式，而非泛化的 Python 语法。
  2. 手动训练循环（Accelerate）：灵活控制每一步的前向/反向传播、梯度裁剪、日志记录。
  3. 参数分组 weight decay：bias 和 LayerNorm 权重不施加 L2 正则，符合 GPT 训练最佳实践。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练设备推荐（GPT-2 124M，context_length=128）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌──────────────────┬────────────┬────────────────────────┬──────────────┐
│ 设备              │ batch_size │ grad_accum_steps       │ mixed_prec   │
├──────────────────┼────────────┼────────────────────────┼──────────────┤
│ 2x H100 NVL 94GB │ 512        │ 2  (eff=512×2×2=2048)  │ bf16         │
│ 1x H100 NVL 94GB │ 512        │ 4  (eff=512×1×4=2048)  │ bf16         │
│ 1x RTX 5090 32GB │ 256        │ 2  (eff=256×1×2=512)   │ bf16 ★ 默认 │
│ 1x A100 40GB     │ 128        │ 8  (eff=128×1×8=1024)  │ bf16         │
│ 2x RTX 4090 24GB │ 32         │ 8  (eff=32×2×8=512)    │ bf16         │
│ 1x RTX 4090 24GB │ 32         │ 16 (eff=32×1×16=512)   │ bf16/fp16    │
│ 1x RTX 3090 24GB │ 16         │ 32 (eff=16×1×32=512)   │ fp16         │
└──────────────────┴────────────┴────────────────────────┴──────────────┘
目标有效 batch size ≥ 512，过小会导致梯度噪声大，收敛不稳定。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
依赖安装（首次运行前执行）：
  pip install datasets transformers[sentencepiece] accelerate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

启动命令：
  单卡：  accelerate launch section-06.train-accelerator.py
  多卡：  accelerate launch --num_processes=2 section-06.train-accelerator.py
  或使用 torchrun（与 accelerate launch 等价）：
          torchrun --nproc_per_node=2 section-06.train-accelerator.py
"""

import os

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, get_full_repo_name, login
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    get_scheduler,
)

from accelerate import Accelerator

# ──────────────────────────────────────────────────────────────────────────────
# 训练超参数（根据上方设备推荐表修改这里）
# ──────────────────────────────────────────────────────────────────────────────
CONTEXT_LENGTH = 128          # 每个训练样本的 token 长度
BATCH_SIZE = 256              # 每卡 batch size（1x RTX 5090 32GB 默认配置）
GRAD_ACCUM_STEPS = 2          # 梯度累积步数；有效 batch = BATCH_SIZE × num_gpus × GRAD_ACCUM_STEPS
NUM_TRAIN_EPOCHS = 1          # 训练轮数（codeparrot-ds 数据量大，1 epoch 约 3.3 万步）
LEARNING_RATE = 5e-4          # 峰值学习率（AdamW + cosine scheduler）
WEIGHT_DECAY = 0.1            # L2 正则系数（仅对非 bias/LayerNorm 参数生效）
WARMUP_STEPS = 1_000          # 学习率线性预热步数
EVAL_STEPS = 5_000            # 每隔多少"优化步"做一次验证（eval + save + push）
LOG_STEPS = 100               # 每隔多少"原始步"打印一次训练 loss
KEYTOKEN_ALPHA = 1.0          # 关键 token 加权系数（alpha=1 表示最多 2× 权重）
OUTPUT_DIR = "codeparrot-ds-accelerate"  # 本地 checkpoint 保存目录

# ──────────────────────────────────────────────────────────────────────────────
# 初始化 Accelerator
#
# mixed_precision="bf16"：启用 BF16 混合精度训练。
#   BF16 vs FP16：BF16 数值范围与 FP32 相同（8 位指数），不易溢出，
#   无需 loss scaling，训练更稳定。H100/A100/RTX 40 系均原生支持 BF16。
#   如果是 RTX 30 系或 V100，改用 mixed_precision="fp16"。
#
# Accelerator 会自动处理：
#   - 模型/优化器/数据的多卡分发（DDP）
#   - 混合精度的自动 cast 和 loss scaling
#   - 主进程判断（is_main_process）
# ──────────────────────────────────────────────────────────────────────────────
accelerator = Accelerator(mixed_precision="bf16")

# ──────────────────────────────────────────────────────────────────────────────
# 登录 HuggingFace Hub（用于推送 checkpoint）
# 通过环境变量 HF_TOKEN 传入，避免交互式输入
# ──────────────────────────────────────────────────────────────────────────────
login(token=os.environ["HF_TOKEN"])

# ──────────────────────────────────────────────────────────────────────────────
# 1. 加载数据集
#
# codeparrot-ds 是从 GitHub 爬取的 Python 代码，经过关键词过滤，
# 只保留包含 pandas/sklearn/matplotlib/seaborn 的文件。
# 训练集约 60 万原始文件，验证集约 3000 条。
# ──────────────────────────────────────────────────────────────────────────────
accelerator.print("加载数据集...")
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict({"train": ds_train, "valid": ds_valid})

# ──────────────────────────────────────────────────────────────────────────────
# 2. 加载 Tokenizer
#
# code-search-net-tokenizer：在 CodeSearchNet 数据集上训练的 BPE tokenizer，
# 词表专门针对代码优化，对 Python 符号（缩进、运算符）有更好的分词效果。
# ──────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 没有 pad token，复用 eos_token

# ──────────────────────────────────────────────────────────────────────────────
# 3. Tokenize 数据集
#
# 策略：将每个文件切分为固定长度 CONTEXT_LENGTH 的块，丢弃尾部不足长的块。
#   - return_overflowing_tokens=True：长文本切块而非截断，最大化利用训练数据
#   - return_length=True：返回每块实际 token 数，用于过滤不完整块
#   - 只保留长度 == CONTEXT_LENGTH 的块，确保所有样本等长（无需 padding）
#
# 处理后训练集约 1670 万条 128-token 样本（原 60 万文件展开后）。
# ──────────────────────────────────────────────────────────────────────────────
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True,   # 长文本切块
        return_length=True,               # 返回每块长度
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == CONTEXT_LENGTH:      # 丢弃尾部不完整块
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


accelerator.print("Tokenize 数据集（多进程并行）...")
tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    batch_size=10000,
    num_proc=10,   # 12 vCPU，留 2 个给 OS 和主进程
)
tokenized_datasets.set_format("torch")   # 将 numpy array 转为 torch.Tensor
accelerator.print(f"训练集大小：{len(tokenized_datasets['train'])} 条样本")

# ──────────────────────────────────────────────────────────────────────────────
# 4. DataLoader
#
# DataCollatorForLanguageModeling 负责：
#   - 将 input_ids 列表堆叠为 batch tensor
#   - 自动生成 labels（与 input_ids 相同，因果 LM 目标）
#   - mlm=False：CLM 模式（预测下一 token），而非 BERT 的 MLM
#
# 注意：这里 batch_size 是单卡的 batch size，DataLoader 不感知多卡。
# Accelerator.prepare() 会负责在多卡间分发数据。
# ──────────────────────────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=4,          # 12 vCPU 单卡，4 个 worker 预加载足够
    pin_memory=True,        # 锁页内存，加速 CPU→GPU 数据传输
)
eval_dataloader = DataLoader(
    tokenized_datasets["valid"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=4,
    pin_memory=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 5. 初始化模型
#
# 从头随机初始化 GPT-2，不加载预训练权重。
# 关键配置参数：
#   vocab_size：使用自定义 tokenizer 的词表大小（~32K），与原始 GPT-2 不同
#   n_ctx：上下文长度，必须与 CONTEXT_LENGTH 一致
# ──────────────────────────────────────────────────────────────────────────────
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=CONTEXT_LENGTH,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
accelerator.print(f"GPT-2 模型参数量：{model_size / 1e6:.1f}M")

# ──────────────────────────────────────────────────────────────────────────────
# 6. 关键 Token 列表（用于自定义 Loss 加权）
#
# 这些 token 代表数据科学库的核心操作。模型对包含这些 token 的样本
# 施加更高权重，使其在学习这些模式时梯度更强，从而更快掌握这些 API 的用法。
#
# 只保留能被单个 token 表示的关键词（多 token 的词加权意义不大）。
# ──────────────────────────────────────────────────────────────────────────────
keytoken_ids = []
for keyword in ["plt", "pd", "sk", "fit", "predict", " plt", " pd", " sk", " fit", " predict"]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        accelerator.print(f"  跳过多 token 关键词：{keyword!r}")
accelerator.print(f"关键 token 数量：{len(keytoken_ids)}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. 自定义加权 Loss 函数
#
# 标准 CLM loss 对所有 token 等权平均。本函数在此基础上：
#   1. 计算每个样本的平均 token loss（逐样本而非全局平均）
#   2. 统计每个样本中关键 token 的出现次数，计算加权系数
#      weight = alpha * (1 + keytoken_count_in_sample)
#      → 不含关键 token 的样本：weight = alpha * 1（最低权重）
#      → 含 N 个关键 token 的样本：weight = alpha * (1 + N)
#   3. 用加权系数对样本 loss 加权平均
#
# 参数说明：
#   inputs：原始输入 token ids，shape=(batch, seq_len)
#   logits：模型输出，shape=(batch, seq_len, vocab_size)
#   keytoken_ids：关键 token 的 id 列表
#   alpha：加权基础系数，越大关键 token 的影响越显著
# ──────────────────────────────────────────────────────────────────────────────
def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=KEYTOKEN_ALPHA):
    # CLM 的 label shift：token[i] 预测 token[i+1]
    # shift_labels: (batch, seq_len-1)
    # shift_logits: (batch, seq_len-1, vocab_size)
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # reduce=False：不自动对 loss 求均值，保留每个 token 的 loss
    # loss shape: (batch * (seq_len-1),)
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # 将 loss reshape 回 (batch, seq_len-1) 并对 seq 维度取均值
    # loss_per_sample shape: (batch,)
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(dim=1)

    # 统计每个样本中关键 token 的总出现次数
    # inputs shape: (batch, seq_len)，kt 为单个关键 token id
    # (inputs == kt).float() shape: (batch, seq_len)
    # .sum(axis=[0, 2]) 在 keytoken 维和 seq 维求和 → (batch,)
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    # weight 最小为 alpha（无关键词时），每多一个关键词 +alpha
    weights = alpha * (1.0 + weights)

    # 加权平均（batch 维度）
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss


# ──────────────────────────────────────────────────────────────────────────────
# 8. 参数分组（weight decay 分组）
#
# GPT 训练最佳实践：
#   - 线性层权重、Embedding 权重：施加 weight_decay（L2 正则，防过拟合）
#   - bias、LayerNorm 权重（weight + bias）：不施加 weight_decay
#     原因：这些参数是"偏移量/缩放量"，正则化会干扰归一化效果
# ──────────────────────────────────────────────────────────────────────────────
def get_grouped_params(model, no_decay=("bias", "LayerNorm.weight")):
    params_with_wd, params_without_wd = [], []
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)
    return [
        {"params": params_with_wd, "weight_decay": WEIGHT_DECAY},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 9. 优化器与学习率调度器
#
# AdamW：Adam + 解耦 weight decay，是 Transformer 训练的标准选择。
#
# cosine scheduler（替代 notebook 中的 linear）：
#   - 学习率从 0 线性预热到 LEARNING_RATE（warmup 阶段）
#   - 然后按余弦曲线从 LEARNING_RATE 衰减到 0（训练更稳定，最终 loss 更低）
#   - linear scheduler 在长训练中后期 lr 仍较高，容易振荡
# ──────────────────────────────────────────────────────────────────────────────
optimizer = AdamW(get_grouped_params(model), lr=LEARNING_RATE)

num_update_steps_per_epoch = len(train_dataloader) // GRAD_ACCUM_STEPS
num_training_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="cosine",             # 余弦退火（比 notebook 原版 linear 更优）
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps,
)

accelerator.print(
    f"训练配置：epochs={NUM_TRAIN_EPOCHS}, "
    f"total_steps={num_training_steps}, "
    f"有效 batch={BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM_STEPS}"
)

# ──────────────────────────────────────────────────────────────────────────────
# 10. Accelerator 准备（多卡分发的核心步骤）
#
# accelerator.prepare() 会自动：
#   - 将 model 用 DDP（DistributedDataParallel）包裹，分发到每张卡
#   - 将 optimizer 适配混合精度（bf16/fp16 的 loss scaling）
#   - 将 dataloader 按进程数分片（每卡看到不同的 batch）
#   - 将 lr_scheduler 适配分布式场景
#
# 注意：prepare() 必须在所有组件创建完毕后一次性调用。
# ──────────────────────────────────────────────────────────────────────────────
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

# ──────────────────────────────────────────────────────────────────────────────
# 11. Hub 仓库初始化（仅主进程执行）
# ──────────────────────────────────────────────────────────────────────────────
if accelerator.is_main_process:
    api = HfApi()
    repo_name = get_full_repo_name(OUTPUT_DIR)
    api.create_repo(repo_name, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator.print(f"Hub 仓库：{repo_name}")

# ──────────────────────────────────────────────────────────────────────────────
# 12. 评估函数
#
# 使用标准 CLM loss（非加权）评估，便于与其他模型对比。
# perplexity = exp(loss)：衡量模型对文本的"困惑程度"，越低越好。
#
# accelerator.gather_for_metrics()：
#   在分布式训练中，每个进程只处理部分验证数据，
#   gather_for_metrics 会收集所有进程的结果并自动去除 padding 样本，
#   相比 gather() 更准确（gather 可能包含 padding 的重复数据）。
# ──────────────────────────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    losses = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        # outputs.loss 是当前进程这批数据的平均 loss（scalar tensor）
        # gather_for_metrics 收集所有进程的 loss，返回 1-D tensor
        loss = accelerator.gather_for_metrics(outputs.loss.unsqueeze(0))
        losses.append(loss)
    # 拼接所有步骤收集到的 loss，计算全局均值
    mean_loss = torch.cat(losses).mean().item()
    try:
        perplexity = torch.exp(torch.tensor(mean_loss)).item()
    except OverflowError:
        perplexity = float("inf")
    return mean_loss, perplexity


# ──────────────────────────────────────────────────────────────────────────────
# 13. 训练前评估基线（随机初始化模型的 loss）
# ──────────────────────────────────────────────────────────────────────────────
eval_loss, perplexity = evaluate()
accelerator.print(f"训练前基线 → loss: {eval_loss:.4f}, perplexity: {perplexity:.1f}")

# ──────────────────────────────────────────────────────────────────────────────
# 14. 训练循环
#
# 梯度累积（GRAD_ACCUM_STEPS）：
#   将多个小 batch 的梯度累加后再更新参数，等价于使用更大的 batch size，
#   同时不增加显存占用。仅在累积满 GRAD_ACCUM_STEPS 步时才执行：
#     - gradient clip（防止梯度爆炸）
#     - optimizer.step()（参数更新）
#     - lr_scheduler.step()（调整学习率）
#     - optimizer.zero_grad()（清零梯度）
#
# accelerator.accumulate(model)（替代 notebook 的手动判断）：
#   自动处理梯度累积期间的 no_sync 上下文，在 DDP 中避免不必要的梯度同步，
#   提升多卡训练效率。（等价于手动写 if step % accum == 0: ...）
#
# accelerator.backward(loss)（替代 loss.backward()）：
#   自动处理混合精度的 loss scaling（bf16/fp16 下防止梯度下溢）。
# ──────────────────────────────────────────────────────────────────────────────
model.train()
completed_steps = 0   # 已完成的"优化步"数（每 GRAD_ACCUM_STEPS 个原始步 = 1 优化步）

for epoch in range(NUM_TRAIN_EPOCHS):
    progress_bar = tqdm(
        enumerate(train_dataloader, start=1),
        total=len(train_dataloader),
        disable=not accelerator.is_local_main_process,  # 非主进程不显示进度条
        desc=f"Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS}",
    )

    for step, batch in progress_bar:
        # accelerator.accumulate 会在非累积步时自动禁用 DDP 梯度同步，
        # 减少跨卡通信开销，提升多卡吞吐量
        with accelerator.accumulate(model):
            # 前向传播：只取 logits，不用模型内置 loss（要用自定义 loss）
            logits = model(batch["input_ids"]).logits

            # 计算加权 loss
            loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)

            # 反向传播（accelerator 自动处理混合精度 loss scaling）
            accelerator.backward(loss)

            # 梯度裁剪：将所有参数的梯度范数限制在 1.0 以内，防止梯度爆炸
            # 注意：必须在 optimizer.step() 之前调用
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # 每完成一次实际参数更新（GRAD_ACCUM_STEPS 个原始步），计数 +1
        if accelerator.sync_gradients:
            completed_steps += 1

        # 定期打印训练状态
        if step % LOG_STEPS == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            # loss 乘以 GRAD_ACCUM_STEPS 还原为"未平均"的原始 loss，便于与 Trainer 对比
            accelerator.print(
                f"[step {step:>6} | opt_step {completed_steps:>5}] "
                f"lr={current_lr:.2e}  "
                f"train_loss={loss.item() * GRAD_ACCUM_STEPS:.4f}"
            )

        # 定期做评估、保存 checkpoint、推送到 Hub
        if completed_steps > 0 and completed_steps % EVAL_STEPS == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print(
                f"\n=== 验证 @ opt_step {completed_steps} ===\n"
                f"  eval_loss={eval_loss:.4f}  perplexity={perplexity:.1f}\n"
            )
            model.train()

            # 多卡场景下，等待所有进程完成评估后再保存，避免文件竞争
            accelerator.wait_for_everyone()

            # unwrap_model：获取 DDP 包裹之前的原始模型（用于保存）
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                OUTPUT_DIR,
                save_function=accelerator.save,     # 使用 accelerator.save 确保兼容分布式
                safe_serialization=True,            # 保存为 safetensors 格式（更安全）
            )

            # 只有主进程负责保存 tokenizer 和推送到 Hub
            if accelerator.is_main_process:
                tokenizer.save_pretrained(OUTPUT_DIR)
                api.upload_folder(
                    repo_id=repo_name,
                    folder_path=OUTPUT_DIR,
                    commit_message=f"Training in progress, step {completed_steps}",
                )

# ──────────────────────────────────────────────────────────────────────────────
# 15. 训练结束：最终评估 + 保存 + 推送
# ──────────────────────────────────────────────────────────────────────────────
eval_loss, perplexity = evaluate()
accelerator.print(
    f"\n=== 训练完成 ===\n"
    f"  最终 eval_loss={eval_loss:.4f}  perplexity={perplexity:.1f}\n"
    f"  总优化步数：{completed_steps}"
)

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    OUTPUT_DIR,
    save_function=accelerator.save,
    safe_serialization=True,
)

if accelerator.is_main_process:
    tokenizer.save_pretrained(OUTPUT_DIR)
    api.upload_folder(
        repo_id=repo_name,
        folder_path=OUTPUT_DIR,
        commit_message="Training complete",
    )
    accelerator.print(f"模型已推送至 Hub：{repo_name}")
