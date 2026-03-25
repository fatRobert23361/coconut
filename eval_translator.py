import torch
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from dataset import CoconutTranslatorDataset, CoconutPureLatentDataset
from translator import CoconutTranslator
from translator_v2 import CoconutTranslator as CoconutSoftPromptTranslator

def evaluate_translator(stage_num, model_path, dataset, tokenizer, mode="context_latent"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = CoconutTranslator(hidden_size=768, vocab_size=len(tokenizer), mode=mode).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    mode = 0 if mode == "latent_only" else 1 if mode == "context_only" else 2
    model.eval()

    results = []
    total_bleu = 0
    smoothie = SmoothingFunction().method1 

    num_samples = len(dataset)
    num_eval = min(1000, num_samples)
    print(f"\n开始评估 Stage {stage_num}... (样本数: {num_eval})")
    correct_num = 0

    for i in tqdm(range(num_eval), desc=f"Evaluating Stage {stage_num}"):
        item = dataset[i]
        

        input_ids = item["input_ids"]
        labels = item["labels"]
        
        context_ids = input_ids[labels == -100]
        target_ids = input_ids[(labels != -100) & (input_ids != tokenizer.pad_token_id)]

        context_text = tokenizer.decode(context_ids, skip_special_tokens=True).strip()
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        
        # --- 核心逻辑修改点 2: 准备向量和掩码 ---
        latent_vec = item["latent_states"].unsqueeze(0).to(device) # (1, 3, 768)
        # 如果你训练时用了 latent_mask，推理时也建议传入（虽然 batch_size=1 时全1即可）
        
        generated_ids = model.translate(latent_vec, context_text, tokenizer, max_new_tokens=40)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # 计算 BLEU
        reference = [target_text.split()]
        candidate = generated_text.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        total_bleu += score

        results.append({
            "context": context_text,
            "target": target_text,
            "pred": generated_text,
            "bleu": score
        })
        if target_text == generated_text:
            correct_num += 1

    avg_bleu = total_bleu / num_eval
    accuracy = correct_num / num_eval
    print(f"\nStage {stage_num} Average BLEU: {avg_bleu:.4f}, Accuracy: {accuracy:.4f}")
    
    # 打印前 3 条结果
    print("-" * 30)
    for i in range(min(10, len(results))):
        print(f"\nExample {i+1}:")
        print(f"Context      : {results[i]['context']}")
        print(f"Ground Truth : {results[i]['target']}")
        print(f"Translated   : {results[i]['pred']}")
        print(f"Single BLEU  : {results[i]['bleu']:.4f}")
    print("-" * 30)
    
    return avg_bleu, accuracy

def evaluate_pure_latent_translator(model_path, dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型 (注意：使用 SoftPrompt 架构)
    model = CoconutSoftPromptTranslator(hidden_size=768).to(device)
    # 加载保存的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    total_bleu = 0
    smoothie = SmoothingFunction().method1 
    correct_num = 0

    num_eval = min(1000, len(dataset))
    print(f"\n开始评估 Pure Latent Stage... (样本数: {num_eval})")

    for i in tqdm(range(num_eval), desc="Evaluating"):
        item = dataset[i]
        
        # 1. 准备 Latent 向量
        # [768] -> [1, 768]
        latent_vec = item["latent_states"].unsqueeze(0).to(device) 
        
        # 2. 准备 Ground Truth 文本用于对比
        # 过滤掉 -100 的 labels 得到原始 target_ids
        target_ids = item["labels"][item["labels"] != -100]
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        
        # 3. 推理生成 (仅依靠向量)
        # 调用新版 translate，内部自动处理 BOS+Start+Latent+End 拼接
        generated_ids = model.translate(latent_vec, max_new_tokens=40)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # 4. 计算指标
        reference = [target_text.split()]
        candidate = generated_text.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        total_bleu += score

        if target_text == generated_text:
            correct_num += 1

        results.append({
            "target": target_text,
            "pred": generated_text,
            "bleu": score
        })

    avg_bleu = total_bleu / num_eval
    accuracy = correct_num / num_eval
    print(f"\nEvaluation Results:")
    print(f"Average BLEU: {avg_bleu:.4f}, Accuracy: {accuracy:.4f}")
    
    # 打印前 10 条对比结果
    print("-" * 30)
    for i in range(min(10, len(results))):
        print(f"\nExample {i+1}:")
        print(f"Ground Truth : {results[i]['target']}")
        print(f"Translated   : {results[i]['pred']}")
        print(f"Single BLEU  : {results[i]['bleu']:.4f}")
    print("-" * 30)
    
    return avg_bleu, accuracy

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def evaluate_context_latent_translator(model_path, dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型 (注意：传入 len(tokenizer) 避免之前的 TypeError)
    # 假设你的类名还是 CoconutTranslator
    from translator_v2 import CoconutTranslator 
    model = CoconutSoftPromptTranslator(hidden_size=768).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    total_bleu = 0
    smoothie = SmoothingFunction().method1 
    correct_num = 0

    num_eval = min(500, len(dataset)) # 评估 500 条足够看清趋势
    print(f"\n开始评估 Context + Latent 模式... (样本数: {num_eval})")

    for i in tqdm(range(num_eval), desc="Evaluating"):
        item = dataset[i]
        
        # 1. 准备输入 (增加 context_ids)
        latent_vec = item["latent_states"].unsqueeze(0).to(device) 
        context_ids = item["context_ids"].unsqueeze(0).to(device)
        
        # 2. 准备 Ground Truth 文本
        target_ids = item["labels"][item["labels"] != -100]
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        
        # 3. 推理生成 (必须传入 context_ids)
        # 调用新版 translate: [Context] + [Start] + [Latent] + [End] -> 生成
        generated_ids = model.translate(latent_vec, context_ids, max_new_tokens=40)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # 4. 计算指标
        reference = [target_text.split()]
        candidate = generated_text.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        total_bleu += score

        if target_text == generated_text:
            correct_num += 1

        # 记录结果，增加 context 文本用于观察
        context_text = tokenizer.decode(item["context_ids"], skip_special_tokens=True).strip()
        results.append({
            "context": context_text,
            "target": target_text,
            "pred": generated_text,
            "bleu": score
        })

    avg_bleu = total_bleu / num_eval
    accuracy = correct_num / num_eval
    print(f"\nEvaluation Results (Context+Latent):")
    print(f"Average BLEU: {avg_bleu:.4f}, Accuracy: {accuracy:.4f}")
    
    # 打印前 10 条对比结果
    print("-" * 30)
    for i in range(min(10, len(results))):
        print(f"\nExample {i+1}:")
        print(f"Context      : {results[i]['context'][:100]}...") # 只打印前100字符
        print(f"Ground Truth : {results[i]['target']}")
        print(f"Translated   : {results[i]['pred']}")
        print(f"Single BLEU  : {results[i]['bleu']:.4f}")
    print("-" * 30)
    
    return avg_bleu, accuracy

import torch
import random
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def run_intervention_study(model_path, dataset, tokenizer, num_samples=500, mode="context_latent"):
    """
    针对 Cross-Attention 版 CoconutTranslator 的干预实验
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoconutTranslator(hidden_size=768, vocab_size=len(tokenizer), mode=mode).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    mode = 0 if mode == "latent_only" else 1 if mode == "context_only" else 2
    model.eval()
    smoothie = SmoothingFunction().method1
    
    # 统计指标
    stats = {
        "normal_acc": 0,
        "swapped_acc": 0,
        "normal_bleu": 0,
        "swapped_bleu": 0
    }
    
    # 结果详情，用于后续分析
    results_detail = []

    print(f"开始干预实验 (Mode: {model.mode})... 样本数: {num_samples}")

    for i in tqdm(range(num_samples)):
        # 1. 获取原始数据项
        item = dataset[i]
        # context_ids = item["context_ids"]
        # 假设你的 dataset 返回的 context_ids 是已经编码好的，我们需要还原回文本传给 translate
        # context_text = tokenizer.decode(context_ids, skip_special_tokens=True).strip()
        context_text = tokenizer.decode(item["input_ids"], skip_special_tokens=True).strip()
        input_ids = item["input_ids"]
        labels = item["labels"]
        target_ids = input_ids[(labels != -100) & (input_ids != tokenizer.pad_token_id)]
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        
        # 2. 正常推理 (Original Latent)
        latent_real = item["latent_states"].unsqueeze(0).to(device) # (1, k, 768)
        
        gen_ids_normal = model.translate(latent_real, context_text, tokenizer)
        pred_normal = tokenizer.decode(gen_ids_normal[0], skip_special_tokens=True).strip()
        
        # 3. 干预推理 (Swapped Latent)
        # 随机抽取一个其他样本的 Latent
        rand_idx = random.choice([idx for idx in range(len(dataset)) if idx != i])
        latent_swapped = dataset[rand_idx]["latent_states"].unsqueeze(0).to(device)
        
        gen_ids_swapped = model.translate(latent_swapped, context_text, tokenizer)
        pred_swapped = tokenizer.decode(gen_ids_swapped[0], skip_special_tokens=True).strip()
        
        # 4. 计算指标
        # 准确率 (Top-1 Exact Match)
        if pred_normal == target_text: stats["normal_acc"] += 1
        if pred_swapped == target_text: stats["swapped_acc"] += 1
        
        # BLEU 分数
        ref = [target_text.split()]
        stats["normal_bleu"] += sentence_bleu(ref, pred_normal.split(), smoothing_function=smoothie)
        stats["swapped_bleu"] += sentence_bleu(ref, pred_swapped.split(), smoothing_function=smoothie)

        results_detail.append({
            "context": context_text,
            "truth": target_text,
            "pred_normal": pred_normal,
            "pred_swapped": pred_swapped
        })

    # 计算平均值
    for key in stats: stats[key] /= num_samples

    print("\n" + "="*50)
    print("干预实验报告 (Intervention Study Report)")
    print("-" * 50)
    print(f"正常模式准确率 (Normal Acc)  : {stats['normal_acc']:.4%}")
    print(f"干扰模式准确率 (Swapped Acc) : {stats['swapped_acc']:.4%}")
    print(f"准确率下降幅度 (Delta Acc)    : {(stats['normal_acc'] - stats['swapped_acc']):.4%}")
    print("-" * 50)
    print(f"正常模式 BLEU (Normal BLEU)  : {stats['normal_bleu']:.4f}")
    print(f"干扰模式 BLEU (Swapped BLEU) : {stats['swapped_bleu']:.4f}")
    print("="*50)

    # 打印前 3 个样本的对比，直观观察
    for i in range(3):
        res = results_detail[i]
        print(f"\n[Example {i+1}]")
        print(f"Context: {res['context'][:80]}...")
        print(f"Truth  : {res['truth']}")
        print(f"Normal : {res['pred_normal']}")
        print(f"Swapped: {res['pred_swapped']}")
        
    return stats

import torch
import random
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# 假设你的模型类定义在 translator.py 中
from translator import CoconutTranslator

def evaluate_intervention2(stage_num, model_path, dataset, tokenizer, mode="context_latent"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = CoconutTranslator(hidden_size=768, vocab_size=len(tokenizer), mode=mode).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    smoothie = SmoothingFunction().method1 

    num_samples = len(dataset)
    num_eval = min(1000, num_samples)
    
    # 统计项
    correct_normal = 0
    correct_swapped = 0
    total_bleu_normal = 0
    total_bleu_swapped = 0

    print(f"\n开始 Stage {stage_num} 干预实验评估... (样本数: {num_eval})")

    for i in tqdm(range(num_eval), desc=f"Intervention Stage {stage_num}"):
        item = dataset[i]
        
        # --- 数据准备 ---
        input_ids = item["input_ids"]
        labels = item["labels"]
        
        # 提取 Context 和 Ground Truth
        context_ids = input_ids[labels == -100]
        target_ids = input_ids[(labels != -100) & (input_ids != tokenizer.pad_token_id)]
        context_text = tokenizer.decode(context_ids, skip_special_tokens=True).strip()
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        
        # --- 1. 正常推理 (使用配对的 Latent) ---
        latent_real = item["latent_states"].unsqueeze(0).to(device)
        gen_ids_normal = model.translate(latent_real, context_text, tokenizer, max_new_tokens=40)
        pred_normal = tokenizer.decode(gen_ids_normal[0], skip_special_tokens=True).strip()

        # --- 2. 干预推理 (使用随机置换的 Latent) ---
        # 随机抽取一个不同索引的样本作为干扰项
        rand_idx = random.choice([idx for idx in range(num_samples) if idx != i])
        latent_swapped = dataset[rand_idx]["latent_states"].unsqueeze(0).to(device)
        
        gen_ids_swapped = model.translate(latent_swapped, context_text, tokenizer, max_new_tokens=40)
        pred_swapped = tokenizer.decode(gen_ids_swapped[0], skip_special_tokens=True).strip()

        # --- 指标计算 ---
        ref = [target_text.split()]
        
        # 正常指标
        bleu_n = sentence_bleu(ref, pred_normal.split(), smoothing_function=smoothie)
        total_bleu_normal += bleu_n
        if pred_normal == target_text: correct_normal += 1
        
        # 干扰指标
        bleu_s = sentence_bleu(ref, pred_swapped.split(), smoothing_function=smoothie)
        total_bleu_swapped += bleu_s
        if pred_swapped == target_text: correct_swapped += 1

        # 保存结果以便打印
        results.append({
            "context": context_text,
            "target": target_text,
            "pred_normal": pred_normal,
            "pred_swapped": pred_swapped,
            "bleu_n": bleu_n,
            "bleu_s": bleu_s
        })

    # --- 打印实验报告 ---
    avg_acc_n = correct_normal / num_eval
    avg_acc_s = correct_swapped / num_eval
    avg_bleu_n = total_bleu_normal / num_eval
    avg_bleu_s = total_bleu_swapped / num_eval
    
    print("\n" + "="*60)
    print(f"Stage {stage_num} 干预实验结果汇总")
    print("-" * 60)
    print(f"正常情况 (Normal)  | Accuracy: {avg_acc_n:.4f}, BLEU: {avg_bleu_n:.4f}")
    print(f"干扰情况 (Swapped) | Accuracy: {avg_acc_s:.4f}, BLEU: {avg_bleu_s:.4f}")
    print(f"下降幅度 (Delta)   | Accuracy Drop: {(avg_acc_n - avg_acc_s):.4f}")
    print("=" * 60)
    
    # 打印对比示例
    print("\n前 5 条详细对比示例:")
    for i in range(min(5, len(results))):
        print(f"\nExample {i+1}:")
        print(f"Context      : {results[i]['context'][:100]}...")
        print(f"Ground Truth : {results[i]['target']}")
        print(f"Normal Pred  : {results[i]['pred_normal']} (BLEU: {results[i]['bleu_n']:.4f})")
        print(f"Swapped Pred : {results[i]['pred_swapped']} (BLEU: {results[i]['bleu_s']:.4f})")
    print("-" * 30)
    
    return avg_acc_n, avg_acc_s

if __name__ == "__main__":
    # 配置与你的脚本保持一致
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    stage = 1
    data_path = f"/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2_context_test/s{stage}_combined.pt"
    model_path = f"/home/haoyang/haoyang/coconut/translator_models/context_latent_v1/translator_gpt2_prosqa_s1/translator_s1_epoch14.pt"
    
    dataset = CoconutTranslatorDataset(data_path, tokenizer, max_text_len=512) 
    evaluate_intervention2(stage, model_path, dataset, tokenizer)
    # evaluate_translator(stage, model_path, dataset, tokenizer, mode="context_latent")

# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     tokenizer.model_max_length = 1024
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # 路径配置
#     stage = 1
#     data_path = f"/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2_context/s{stage}_combined.pt"
#     model_path = f"/home/haoyang/haoyang/coconut/translator_models/context_latent_v1/translator_gpt2_prosqa_s1/translator_s1_epoch14.pt"
    
#     dataset = CoconutTranslatorDataset(data_path, tokenizer, max_text_len=512) 
#     # evaluate_translator(stage, model_path, dataset, tokenizer)
#     run_intervention_study(model_path, dataset, tokenizer)