import torch
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

# ... 保持 get_dataset 原样 ...

@dataclass
class MyCollator:
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        # 1. 基础检查
        assert self.tokenizer.padding_side == "right"
        
        # 2. 处理 Coconut 特有的 Position ID 偏移（对齐 Latent 开启 KV Cache 复用）
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(self.latent_id)
                else:
                    n_tok_pad = 0
                
                feature["position_ids"] = [0] * n_tok_pad + list(range(len(feature["input_ids"])))
                feature["input_ids"] = [self.tokenizer.pad_token_id] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature["labels"]
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        # 3. 提取并处理 translator_labels (新增部分)
        # 期望形状: (batch, max_n_latents, max_step_len)
        has_translator_data = "translator_step_tokens" in features[0]
        batch_translator_labels = None
        
        if has_translator_data:
            # 找到当前 batch 中最大的 latent 数量和最大的步骤文本长度
            max_latents_in_batch = max(len(f["translator_step_tokens"]) for f in features)
            
            # 这里的逻辑是将每个 step 的 tokens 填充对齐
            # 注意：如果 c_thought > 1，一个 step 对应多个 latent，
            # 我们这里简单处理：将文本赋给该 step 对应的最后一个 latent token 位置
            temp_batch_translator = []
            for feature in features:
                steps_tokens = feature["translator_step_tokens"] # List[List[int]]
                
                # 填充步骤数量到 max_latents_in_batch
                padded_steps = steps_tokens + [[]] * (max_latents_in_batch - len(steps_tokens))
                
                # 对每一个 step 内部进行 padding
                # (这里可以根据需要决定是否在 Collator 里 pad，或者在 forward 里处理)
                temp_batch_translator.append(padded_steps)
            
            # 将其转换为 Tensor (Batch, Latents, SeqLen)
            # 先手动对齐 SeqLen
            max_step_len = 0
            for b in temp_batch_translator:
                for s in b:
                    max_step_len = max(max_step_len, len(s))
            
            max_step_len = max(max_step_len, 2)
            
            # --- 关键修复 2：安全获取 pad_id ---
            # GPT-2 默认没有 pad_token，如果不做 fallback 会传入 None 导致 torch.full 报错
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            # 构造最终 tensor (使用 pad_id 而不是 self.tokenizer.pad_token_id)
            final_t_labels = torch.full(
                (len(features), max_latents_in_batch, max_step_len), 
                pad_id, 
                dtype=torch.long
            )
            
            for b_idx, b_data in enumerate(temp_batch_translator):
                for l_idx, s_data in enumerate(b_data):
                    if len(s_data) > 0:
                        final_t_labels[b_idx, l_idx, :len(s_data)] = torch.tensor(s_data)
            
            batch_translator_labels = final_t_labels

        # 4. 常规 Padding (input_ids, labels, attention_mask)
        return_tensors = "pt"
        label_name = "label" if "label" in features[0].keys() else "labels"
        non_label_position_features = [{k: v for k, v in f.items() if k not in [label_name, "position_ids", "translator_step_tokens"]} for f in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, non_label_position_features, padding=True, return_tensors=return_tensors
        )

        # 手动处理 labels 和 position_ids
        for key in [label_name, "position_ids"]:
            if key in features[0]:
                data = [feature[key] for feature in features]
                max_l = max(len(d) for d in data)
                pad_val = self.label_pad_token_id if key == label_name else 0
                batch[key] = torch.tensor([d + [pad_val] * (max_l - len(d)) for d in data], dtype=torch.int64)

        if batch_translator_labels is not None:
            batch["translator_labels"] = batch_translator_labels

        return batch


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
    eos_id=None
):
    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        # 1. 决定当前样本训练到第几个阶段 (Latent Stage)
        if random.random() < configs.uniform_prob:
            scheduled_stage_to_train = random.choice(list(range(len(sample["steps_tokenized"]) + 1)))
        else:
            scheduled_stage_to_train = scheduled_stage

        # 限制最大 latent 阶段
        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = len(sample["steps_tokenized"]) # 如果超出，可能全部转 latent (视逻辑而定)
            n_latent_stages = min(len(sample["steps_tokenized"]), configs.max_latent_stage)
        else:
            n_skip_steps = scheduled_stage_to_train
            n_latent_stages = scheduled_stage_to_train

        if configs.no_cot:
            n_skip_steps = 100
            n_latent_stages = 0

        # 总的 latent token 数量
        total_latent_tokens = n_latent_stages * configs.c_thought

        # --- 新增：提取被替换掉的思维步骤作为翻译器的监督信号 ---
        # 我们按步骤存储。如果 c_thought > 1，我们会把步骤文本分配给对应的 latent 块
        translator_step_tokens = []
        for step_idx in range(n_latent_stages):
            # 如果 c_thought > 1，每组 latent 的前 c_thought-1 个都是空（无监督）
            for _ in range(configs.c_thought - 1):
                translator_step_tokens.append([])

            if step_idx < len(sample["steps_tokenized"]):
                # 有对应的真实推理步骤：加 eos_id 作为结束符，作为翻译器的监督目标
                step_text = sample["steps_tokenized"][step_idx]
                translator_step_tokens.append(step_text + [eos_id])
            else:
                # 额外的 latent（题目步骤数 < n_latent_stages）：
                # 不附加任何 token，Collator 会填充 pad，forward 中会被 mask 为 -100，
                # 不产生监督信号。避免依赖 pad_id == eos_id 的侥幸巧合。
                translator_step_tokens.append([])
        # -----------------------------------------------------

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * total_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:]))
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100] * (len(sample["question_tokenized"]) + total_latent_tokens + n_additional_tokens)
            + tokens[total_latent_tokens + n_additional_tokens + len(sample["question_tokenized"]) :],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "translator_step_tokens": translator_step_tokens, # 传递给 Collator
        }

    # ... 保持分布式广播逻辑原样 ...
    # (使用 dataset.map 调用 process_dataset)
    return base_dataset.map(process_dataset, remove_columns=list(base_dataset.features), num_proc=32)

def get_question_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    """
    专门用于生成 (Generation) 评估的数据集。
    它会截断真实答案，只返回 [问题] + [潜变量占位符]。
    """
    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        # 这里的阶段计算与训练集保持一致
        if random.random() < getattr(configs, "uniform_prob", 0.0):
            scheduled_stage_to_train = random.choice(list(range(len(sample["steps_tokenized"]) + 1)))
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_latent_stages = min(len(sample["steps_tokenized"]), configs.max_latent_stage)
        else:
            n_latent_stages = min(scheduled_stage_to_train, len(sample["steps_tokenized"]))

        if getattr(configs, "no_cot", False):
            n_latent_stages = 0

        total_latent_tokens = n_latent_stages * configs.c_thought

        # 核心改动：丢弃所有的文字思维和答案，只保留前缀
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * total_latent_tokens
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "labels": tokens,  # 生成时 labels 不起作用，占位即可
            "attention_mask": [1] * len(tokens),
            "idx": sample.get("idx", 0),
            "position_ids": list(range(len(tokens))),
        }

    # 保留 answer 和 steps 字段供后续计算准确率使用
    columns_to_remove = [col for col in base_dataset.features.keys() if col not in ["answer", "steps"]]
    return base_dataset.map(process_dataset, remove_columns=columns_to_remove, num_proc=32)