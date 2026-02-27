import torch
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from dataset import CoconutTranslatorDataset

def evaluate_translator(stage_num, model_path, dataset, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    from translator import CoconutTranslator # 确保导入路径正确
    model = CoconutTranslator(hidden_size=768, vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []
    total_bleu = 0

    print(f"开始评估 Stage {stage_num}...")
    # 为了评估效率，可以只抽样 500-1000 条
    # eval_samples = dataset[:1000] if len(dataset) > 1000 else dataset
    num_samples = len(dataset)
    num_eval = min(1000, num_samples)
    print(f"检测到总样本数: {num_samples}, 将评估前 {num_eval} 条。")
    for i in tqdm(range(num_eval), desc=f"Evaluating Stage {stage_num}"):
        item = dataset[i]   
        latent_vec = item["latent_states"].unsqueeze(0).to(device) # (1, 3, 768)
        target_text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        
        # 使用你定义的 translate 函数进行生成
        generated_ids = model.translate(latent_vec, tokenizer, max_new_tokens=40)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 计算 BLEU
        reference = [target_text.split()]
        candidate = generated_text.split()
        score = sentence_bleu(reference, candidate)
        total_bleu += score

        results.append({
            "target": target_text,
            "pred": generated_text,
            "bleu": score
        })

    avg_bleu = total_bleu / num_eval
    print(f"Stage {stage_num} Average BLEU: {avg_bleu:.4f}")
    
    # 打印前 3 条结果看看“翻译”得像不像人话
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Ground Truth: {results[i]['target']}")
        print(f"Translated  : {results[i]['pred']}")
    
    return avg_bleu

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # 循环评估所有 Stage 并记录数据画图
    stage_scores = {}
    data_path = f"/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2/s2_combined.pt"
    model_path = f"/home/haoyang/haoyang/coconut/translator_models/translator_s2_best.pt"
        # 加载对应的 Dataset 类
    dataset = CoconutTranslatorDataset(data_path, tokenizer) 
    stage_scores[2] = evaluate_translator(2, model_path, dataset, tokenizer)