import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import os

# 假设你已经定义了 CoconutTranslator 和 CoconutTranslatorDataset
from translator import CoconutTranslator
from dataset import CoconutTranslatorDataset, CoconutContextLatentDataset as CoconutTranslatorDatasetV2
from eval_translator import evaluate_translator, evaluate_pure_latent_translator, evaluate_context_latent_translator
import wandb
from translator_v2 import CoconutTranslator as CoconutTranslatorV2

def train_translator_stage(stage_num, data_path, mode="context_latent", save_dir="translator_gpt2_prosqa_s1"):
    # --- 1. 参数配置 ---
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project="coconut_translator_prosqa_latent_as_embedding", 
               group="GPT2_4Layer_Context512",
               name=f"Stage_{stage_num}_Training_{mode}",
               config={
                   "stage": stage_num,
                   "mode": mode,
                   "batch_size": BATCH_SIZE,
                   "epochs": EPOCHS,
                   "learning_rate": LEARNING_RATE,
                   "weight_decay": WEIGHT_DECAY,
                   "warmup_ratio": WARMUP_RATIO
               }
    )
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 这里的 tokenizer 必须与 COCONUT 模型使用的一致
    model = CoconutTranslatorV2(hidden_size=768, mode=mode).to(DEVICE)
    model.load_state_dict(torch.load(f"/home/haoyang/haoyang/coconut/translator_models/latent_as_embedding/translator_gpt2_prosqa_s1_optimizer/translator_s1_epoch28.pt", map_location=DEVICE), strict=False)
    tokenizer = model.tokenizer
    # --- 2. 数据准备 ---
    print(f"正在加载 Stage {stage_num} 的合并数据: {data_path}")
    # 使用之前定义的 Dataset 类处理列表数据
    # dataset = CoconutTranslatorDataset(data_path, tokenizer, max_latent=3, max_text_len=512, mode=mode) 
    dataset = CoconutTranslatorDatasetV2(data_path,  model.tokenizer, max_context_len=512, max_target_len=128)  
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- 3. 初始化翻译器 ---
    # hidden_size 为 768，vocab_size 为 50260
    wandb.watch(model, log="all", log_freq=100)
    
    # 如果是数据量极少的 Stage 6，可以考虑加载 Stage 1 的模型进行微调
    if stage_num == 6:
        s1_path = os.path.join(save_dir, "translator_s1_best.pt")
        if os.path.exists(s1_path):
            print("Stage 6 数据较少，正在加载 Stage 1 权重进行迁移学习...")
            model.load_state_dict(torch.load(s1_path))
            LEARNING_RATE = 1e-5 # 微调时使用更小的学习率

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # optimizer = get_optimizer(model, base_lr=LEARNING_RATE, projector_lr=1e-3, wte_lr=5e-4, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader)*EPOCHS
    warm_up_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)

    # --- 4. 训练循环 ---
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Stage {stage_num} Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # latent_states: (batch, 3, 768)
            # input_ids: (batch, seq_len)
            loss, _ = model(
                latent_states=batch["latent_states"].to(DEVICE),
                context_ids=batch["context_ids"].to(DEVICE),
                input_ids=batch["input_ids"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if batch_idx % 10 == 0:
                avg_loss = running_loss / 10
                running_loss = 0.0
                wandb.log({"train_loss": avg_loss, "epoch": epoch, "learning_rate": scheduler.get_last_lr()[0]})
                
        
        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                v_loss, _ = model(
                    latent_states=batch["latent_states"].to(DEVICE),
                    context_ids=batch["context_ids"].to(DEVICE),
                    input_ids=batch["input_ids"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE)
                )
                total_val_loss += v_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            "average_train_loss_epoch": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch
        })

        # 保存最优模型
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     save_path = os.path.join(save_dir, f"translator_s{stage_num}_best.pt")
        #     torch.save(model.state_dict(), save_path)
        #     print(f"最优模型已保存至: {save_path}")
        
        save_path = os.path.join(save_dir, f"translator_s{stage_num}_epoch{epoch+1}_restart.pt")
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")
        
        avg_bleu, accuracy = evaluate_context_latent_translator(save_path, val_ds, model.tokenizer)
        wandb.log({
            "val_bleu": avg_bleu,
            "val_accuracy": accuracy
        })

    wandb.finish()

import torch.optim as optim

def get_optimizer(model, base_lr=2e-5, projector_lr=1e-3, wte_lr=5e-4, weight_decay=0.01):
    """
    针对 Soft Prompting 架构优化的差异化学习率配置
    - base_lr: 用于预训练的 GPT-2 解码器层 (微调)
    - projector_lr: 用于新增的 Feature Projector (高学习率以拉开相似度)
    - wte_lr: 用于词嵌入层，特别是新加入的特殊 Token
    """
    projector_params = []
    wte_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. 特征放大器模块 (最需要大步快走的随机参数)
        if "projector" in name:
            projector_params.append(param)
        
        # 2. 词嵌入层 (包含新扩容的特殊 Token 权重)
        elif "wte" in name or "transformer.wte" in name:
            wte_params.append(param)
            
        # 3. GPT-2 核心解码层 (已经有预训练知识，适合小步慢走)
        else:
            base_params.append(param)
    # 打印参数分组情况，方便确认是否漏掉了模块
    print(f"优化器分组详情:")
    print(f" - Base Layers: {len(base_params)} params, LR: {base_lr}")
    print(f" - Projector: {len(projector_params)} params, LR: {projector_lr}")
    print(f" - WTE/Embedding: {len(wte_params)} params, LR: {wte_lr}")
    
    optimizer_grouped_parameters = [
        {
            "params": base_params, 
            "lr": base_lr, 
            "weight_decay": weight_decay
        },
        {
            "params": projector_params, 
            "lr": projector_lr, 
            "weight_decay": weight_decay
        },
        {
            "params": wte_params, 
            "lr": wte_lr, 
            "weight_decay": weight_decay
        }
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    return optimizer
if __name__ == "__main__":
    # 示例：先训练数据最充足的 Stage 1
    # train_translator_stage(stage_num=1, data_path="/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2_context/s1_combined.pt")
    # for mode in ["latent_only", "context_only"]:
    for mode in ["latent_only"]:
        for stage in range(1, 7):
            data_path = f"/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2_context/s{stage}_combined.pt"
            train_translator_stage(stage_num=stage, data_path=data_path, mode=mode, save_dir=f"translator_models/latent_as_embedding/translator_gpt2_prosqa_s{stage}_optimizer")