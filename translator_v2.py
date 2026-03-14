import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

class CoconutTranslator(nn.Module):
    def __init__(self, hidden_size=768, mode="context_latent"):
        super().__init__()
        
        # 1. 先准备 Tokenizer，确定最终词表大小
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # 修正拼写：确保前后一致
        special_tokens = {'additional_special_tokens': ['<|start_latent|>', '<|end_latent|>']} 
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. 配置模型：使用 len(self.tokenizer) 动态设置
        self.config = GPT2Config.from_pretrained("gpt2")
        self.config.n_layer = 6
        self.config.n_head = 12
        self.config.vocab_size = len(self.tokenizer) 
        self.config.add_cross_attention = False # 软提示不需要 cross-attention
        
        # 加载预训练权重，并调整 Embedding 层大小
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=self.config, ignore_mismatched_sizes=True)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start_latent|>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|end_latent|>")
        self.bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def forward(self, latent_states, context_ids,input_ids, labels=None, attention_mask=None):
        batch_size = latent_states.shape[0]
        device = latent_states.device
        
        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)
        # 1. 转换特殊 Token 为 Embedding
        # 这里的 [batch_size, 1] 确保了形状对齐
        context_embeds = self.decoder.transformer.wte(context_ids)
        # bos_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.bos_id, device=device))
        start_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.start_id, device=device))
        end_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.end_id, device=device))
        target_embeds = self.decoder.transformer.wte(input_ids)

        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)

        # 2. 拼接序列
        full_embeds = torch.cat([context_embeds, start_embed, latent_states, end_embed, target_embeds], dim=1)

        # 3. 构造 Mask 和 Labels
        # 前面 3 个位置固定为 1 (Start, Latent, End)
        context_mask = (context_ids != self.tokenizer.pad_token_id).long()
        latent_mask = torch.ones((batch_size, 3), device=device)
        full_attention_mask = torch.cat([context_mask, latent_mask, attention_mask], dim=1)

        if labels is not None:
            latent_labels = torch.full((batch_size,context_ids.shape[1] + 3), -100, device=device)
            full_labels = torch.cat([latent_labels, labels], dim=1)
        else:
            full_labels = None

        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels
        )
        return outputs.loss, outputs.logits
    
    @torch.no_grad()
    def translate(self, latent_states, context_ids, max_new_tokens=40):
        """纯 Latent 解码推理"""
        device = latent_states.device
        batch_size = latent_states.shape[0]

        # 构造前缀向量: BOS + Start + Latent + End
        # bos_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.bos_id, device=device))
        context_embeds = self.decoder.transformer.wte(context_ids)
        start_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.start_id, device=device))
        end_embed = self.decoder.transformer.wte(torch.full((batch_size, 1), self.end_id, device=device))
        
        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)
            
        current_embeds = torch.cat([context_embeds, start_embed, latent_states, end_embed], dim=1)
        
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