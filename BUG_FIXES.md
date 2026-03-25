# Bug 修改文档

## 概述

本次共修复 4 个 Bug，涉及 `mixed.py` 和 `train.py` 两个文件。

---

## Bug 1 — `inputs_embeds` 原地修改破坏反向传播

**文件**：`mixed.py`
**位置**：原第 101 行（`forward` 方法内，latent 注入循环中）
**严重性**：🔴 严重 — 导致梯度计算错误，模型实际上无法正常训练

### 问题描述

原代码在 `pass_idx` 迭代循环中，对已参与计算图的张量 `inputs_embeds` 做原地写入：

```python
# 有问题的原代码
inputs_embeds[b_idx, token_idx, :] = latent_vec.squeeze(0).squeeze(0)
```

`inputs_embeds` 由 `self.embedding(input_ids)` 产生，是计算图中的中间节点。对其做原地修改（`__setitem__`）会破坏 autograd 的版本追踪机制，导致反向传播时报错或静默地给出错误梯度，使联合训练实质上失效。

原始 `coconut.py` 中已经用 `tensor_list + torch.stack` 方案规避了这个问题，`mixed.py` 在移植时退回了原地写入。

### 修改方案

移除原地写入，改为收集需要替换的位置 `filling_indices`，在内层循环结束后用非原地的方式重建整个 `inputs_embeds`：

```python
# 修复后：先收集需要替换的位置
filling_indices.append((b_idx, token_idx))

# 内层循环结束后，用 tensor_list + torch.stack 重建张量
if filling_indices:
    tensor_list = [
        [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
        for batch_idx in range(inputs_embeds.shape[0])
    ]
    for (b_idx, token_idx), latent_vec in zip(filling_indices, current_pass_latents):
        tensor_list[b_idx][token_idx] = latent_vec.squeeze(0).squeeze(0)
    inputs_embeds = torch.stack([
        torch.stack(tensor_list[batch_idx])
        for batch_idx in range(inputs_embeds.shape[0])
    ])
```

---

## Bug 2 — Eval 数据集使用了错误的特殊 Token ID

**文件**：`train.py`
**位置**：`evaluate_and_log_wandb` 函数内，`get_question_latent_dataset` 调用处（原第 264、266 行）
**严重性**：🔴 严重 — eval 序列结构与训练时不一致，所有准确率数字不可信

### 问题描述

在计算生成准确率时，构造 eval 数据集使用了错误的边界 token：

```python
# 有问题的原代码
eval_gen_ds = get_question_latent_dataset(
    ...
    start_id=tokenizer.bos_token_id,   # 错误：应为 <|start-latent|> 的 ID
    latent_id=latent_id,
    end_id=tokenizer.eos_token_id      # 错误：应为 <|end-latent|> 的 ID
)
```

训练时序列结构为 `[问题] <|start-latent|> <latent>... <|end-latent|> [答案]`，而 eval 时变成了 `[问题] <bos> <latent>... <eos>`，模型从未见过这种输入格式，导致评估结果完全失真。

注意：同一函数内 validation loss 的计算（`get_cot_latent_dataset` 调用处）已经正确使用了 `convert_tokens_to_ids`，只有 generation accuracy 部分出了问题。

### 修改方案

将 `start_id` 和 `end_id` 作为参数加入 `evaluate_and_log_wandb` 的函数签名，由调用方（`train` 函数）传入正确的 token ID：

```python
# 函数签名修改
def evaluate_and_log_wandb(model, raw_val, tokenizer, stage, epoch, device, cfg, latent_id, start_id, end_id):

# eval 数据集构造修复
eval_gen_ds = get_question_latent_dataset(
    ...
    start_id=start_id,   # 正确：<|start-latent|> 的 ID
    latent_id=latent_id,
    end_id=end_id        # 正确：<|end-latent|> 的 ID
)

# 调用处同步修改
evaluate_and_log_wandb(model, raw_val, tokenizer, stage, epoch, device, cfg, latent_id, start_id, end_id)
```

---

## Bug 3 — Checkpoint 被重复加载两次

**文件**：`train.py`
**位置**：`train` 函数内，checkpoint 加载逻辑（原第 124–142 行）
**严重性**：🟡 中等 — 冗余 I/O，且两次加载使用了不同的 `map_location`，语义不一致

### 问题描述

原代码对同一个 checkpoint 文件加载了两次：

```python
# 第一次：加载模型权重，map_location="cpu"
resume_ckpt = torch.load(cfg["resume_from_checkpoint"], map_location="cpu")
model.load_state_dict(resume_ckpt["model_state_dict"])

# ... 中间穿插了数据加载和优化器初始化 ...

# 第二次：加载优化器状态，map_location=device（与第一次不一致）
checkpoint = torch.load(cfg["resume_from_checkpoint"], map_location=device)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

两次加载使用了不同的 `map_location`：第一次加载到 CPU，第二次加载到训练设备。除了造成双倍 I/O 开销，对于大模型 checkpoint 来说，CPU 内存也会短暂被占用两份。

### 修改方案

在函数作用域内将 `resume_ckpt` 声明为 `None`，只加载一次（统一使用 `map_location=device`），后续直接复用该变量：

```python
resume_ckpt = None
if cfg.get("resume_from_checkpoint"):
    resume_ckpt = torch.load(cfg["resume_from_checkpoint"], map_location=device)
    model.load_state_dict(resume_ckpt["model_state_dict"])

# 优化器初始化后，直接复用已加载的 dict
optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
```

---

## Bug 4 — 联合训练缺少梯度裁剪

**文件**：`train.py`
**位置**：`train` 函数内，训练循环的 `optimizer.step()` 前（原第 191–193 行）
**严重性**：🟠 较严重 — 联合训练的梯度耦合使梯度爆炸风险显著高于单模型训练

### 问题描述

原代码的训练循环中没有梯度裁剪：

```python
if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
    optimizer.step()   # 没有裁剪，梯度可能很大
    optimizer.zero_grad()
```

`CoconutWithTranslator` 的 loss 由两部分组成：`coconut_loss + λ * translator_loss`。翻译器 loss 会通过 latent 向量反向传播进 Coconut 主干网络，形成双向梯度耦合。在早期 stage（尤其 stage 1）翻译器权重几乎随机，产生的梯度信号噪声极大，容易引发梯度爆炸，导致 loss 突然变 NaN 或训练发散。

### 修改方案

在 `optimizer.step()` 之前添加梯度范数裁剪，阈值设为业界常用的 `1.0`：

```python
if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 修改文件汇总

| 文件 | 修改内容 |
|------|---------|
| `mixed.py` | Bug 1：将原地 `inputs_embeds` 写入改为 `tensor_list + torch.stack` 重建方案 |
| `train.py` | Bug 2：`evaluate_and_log_wandb` 函数签名新增 `start_id`、`end_id` 参数，修正 eval 数据集中错误的 token ID |
| `train.py` | Bug 3：checkpoint 由加载两次改为加载一次，统一使用 `map_location=device`，后续复用同一变量 |
| `train.py` | Bug 4：训练循环中在 `optimizer.step()` 前添加 `clip_grad_norm_`，`max_norm=1.0` |
