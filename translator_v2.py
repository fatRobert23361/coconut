import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class CoconutTranslator(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=50260, mode="context_latent"):
        super().__init__()
        # 配置一个轻量级的 GPT-2 作为解码器
        # 必须开启 add_cross_attention 以接收潜状态向量
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=6,
            n_head=12,
            add_cross_attention=False
        )
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=self.config)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        special_tokens = {'additional_special_tokens': ['<|start_latent|>','<end_latent>']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start_latent|>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<end_latent>")
        self.bos_id = self.tokenizer.eos_token_id

    def forward(self, latent_states, latent_mask, input_ids, labels=None, attention_mask=None):
        """
        latent_states: (batch, k, 768) - 这里的 k 是 1, 2 或 3
        target_ids: (batch, seq_len) - 推理步骤的 Token ID
        """
        batch_size = latent_states.shape[0]
        device = latent_states.device

        # 1. 获取基础 Embedding
        # 转换起始符和边界符为 Embedding
        bos_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.bos_id, device=device))
        start_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.start_id, device=device))
        end_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.end_id, device=device))
        # 转换 Target 文本为 Embedding
        target_embeds = self.decoder.transformer.wte(input_ids)

        # 2. 强制确保 latent_states 是 3D: [Batch, 1, 768]
        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)

        # 3. 拼接序列: BOS + Start + Latent + End + Target
        full_embeds = torch.cat([
            bos_embed, start_embed, latent_states, end_embed, target_embeds
        ], dim=1)

        # 4. 构造对应的 Labels (前面 4 个位置设为 -100 不计 Loss)
        latent_fill = torch.full((batch_size, 4), -100, device=device)
        full_labels = torch.cat([latent_fill, labels], dim=1)

        # 5. 构造全量的 Attention Mask
        latent_mask = torch.ones((batch_size, 4), device=device)
        full_attention_mask = torch.cat([latent_mask, attention_mask], dim=1)

        # 6. 使用 inputs_embeds 模式调用
        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels
        )
        return outputs.loss, outputs.logits

    @torch.no_grad()
    def translate(self, latent_states, max_new_tokens=40):
        """纯 Latent 解码推理"""
        device = latent_states.device
        batch_size = latent_states.shape[0]

        # 构造前缀向量: BOS + Start + Latent + End
        bos_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.bos_id, device=device))
        start_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.start_id, device=device))
        end_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.end_id, device=device))
        
        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)
            
        current_embeds = torch.cat([bos_embed, start_embed, latent_states, end_embed], dim=1)
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            outputs = self.decoder(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_ids.append(next_token)
            
            # 转换新生成的 Token 为 Embedding 并拼接
            next_embed = self.decoder.transformer.wte(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return torch.cat(generated_ids, dim=1)