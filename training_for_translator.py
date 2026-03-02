import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# 假设你已经定义了 CoconutTranslator 和 CoconutTranslatorDataset
from translator import CoconutTranslator
from dataset import CoconutTranslatorDataset

def train_translator_stage(stage_num, data_path, save_dir="translator_models"):
    # --- 1. 参数配置 ---
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 5e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 这里的 tokenizer 必须与 COCONUT 模型使用的一致
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1024 
    tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 数据准备 ---
    print(f"正在加载 Stage {stage_num} 的合并数据: {data_path}")
    # 使用之前定义的 Dataset 类处理列表数据
    dataset = CoconutTranslatorDataset(data_path, tokenizer, max_latent=3, max_text_len=512) 
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- 3. 初始化翻译器 ---
    # hidden_size 为 768，vocab_size 为 50260
    model = CoconutTranslator(hidden_size=768, vocab_size=len(tokenizer)).to(DEVICE)
    
    # 如果是数据量极少的 Stage 6，可以考虑加载 Stage 1 的模型进行微调
    if stage_num == 6:
        s1_path = os.path.join(save_dir, "translator_s1_best.pt")
        if os.path.exists(s1_path):
            print("Stage 6 数据较少，正在加载 Stage 1 权重进行迁移学习...")
            model.load_state_dict(torch.load(s1_path))
            LEARNING_RATE = 1e-5 # 微调时使用更小的学习率

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练循环 ---
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Stage {stage_num} Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # latent_states: (batch, 3, 768)
            # input_ids: (batch, seq_len)
            loss, _ = model(
                latent_states=batch["latent_states"].to(DEVICE),
                latent_mask=batch["latent_mask"].to(DEVICE),
                input_ids=batch["input_ids"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                v_loss, _ = model(
                    latent_states=batch["latent_states"].to(DEVICE),
                    latent_mask=batch["latent_mask"].to(DEVICE),
                    input_ids=batch["input_ids"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE)
                )
                total_val_loss += v_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Average Val Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, f"translator_s{stage_num}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"最优模型已保存至: {save_path}")

if __name__ == "__main__":
    # 示例：先训练数据最充足的 Stage 1
    train_translator_stage(stage_num=1, data_path="/home/haoyang/haoyang/coconut/data/coconut_prosqa_gpt2_context/s1_combined.pt")