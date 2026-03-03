import torch
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from dataset import CoconutTranslatorDataset
from translator import CoconutTranslator

def evaluate_translator(stage_num, model_path, dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = CoconutTranslator(hidden_size=768, vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        
        context_text = tokenizer.decode(context_ids, skip_special_tokens=True)
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1024
    tokenizer.pad_token = tokenizer.eos_token
    
    # 路径配置
    stage = 1
    data_path = f"/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2_context/s{stage}_combined.pt"
    model_path = f"/home/haoyang/haoyang/coconut/translator_models/translator_s{stage}_best.pt"
    
    dataset = CoconutTranslatorDataset(data_path, tokenizer, max_text_len=512) 
    evaluate_translator(stage, model_path, dataset, tokenizer)