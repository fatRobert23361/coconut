import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class CoconutTranslator(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=50260):
        super().__init__()
        # 配置一个轻量级的 GPT-2 作为解码器
        # 必须开启 add_cross_attention 以接收潜状态向量
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=4,           # 4层足以证明非线性解码能力
            n_head=8,
            add_cross_attention=True 
        )
        self.decoder = GPT2LMHeadModel(self.config)

    def forward(self, latent_states, target_ids, attention_mask=None):
        """
        latent_states: (batch, k, 768) - 这里的 k 是 1, 2 或 3
        target_ids: (batch, seq_len) - 推理步骤的 Token ID
        """
        # labels 用于计算 CrossEntropy Loss
        outputs = self.decoder(
            input_ids=target_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=latent_states, # 将潜状态作为“记忆”输入
            labels=target_ids
        )
        return outputs.loss, outputs.logits

    @torch.no_grad()
    def translate(self, latent_states, tokenizer, max_new_tokens=30):
        """用于推理：将向量翻译回文字"""
        device = latent_states.device
        # 初始输入为起始符
        generated = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        for _ in range(max_new_tokens):
            outputs = self.decoder(
                input_ids=generated,
                encoder_hidden_states=latent_states
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        return generated