import os
import torch
import json
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from dataset import get_dataset, get_cot_latent_dataset, MyCollator
from utils import Config, set_seed

def extract_and_save():
    # load config
    config_path = "args/prosqa_coconut.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)
    set_seed(configs.seed)
    
    # load the tokenizer
    checkpoint_path = configs.load_model_path 
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # add special tokens for latent representations
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # initialize the Coconut model
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # load the checkpoint
    if checkpoint_path != "None":
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # load the dataset
    with open(configs.train_path) as f:
        raw_train_data = json.load(f)
    
    # get the tokenized dataset
    base_dataset = get_dataset(configs.train_path, tokenizer)
    collator = MyCollator(tokenizer, latent_id=latent_id)

    
    output_dir = "extracted_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # enumerate the inference stages
    for stage in range(1, configs.max_latent_stage + 1):
        # get the dataloader
        curr_dataset = get_cot_latent_dataset(
            stage, base_dataset, configs, start_id, latent_id, end_id
        )
        loader = torch.utils.data.DataLoader(curr_dataset, batch_size=1, collate_fn=collator)

        for i, batch in enumerate(tqdm(loader, desc=f"Stage {stage} Extraction")):
            idx = batch.pop("idx").item()
            if stage - 1 >= len(raw_train_data[idx]["steps"]):
                continue
            target_step_text = raw_train_data[idx]["steps"][stage - 1] 
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch)
                
                # we use c_thought=1 here
                current_latent = outputs.latent_states[stage - 1] 

            # save the extracted latent vector and the corresponding target text, question, and metadata
            data_pair = {
                "latent_vec": current_latent.cpu(),     
                "target_text": target_step_text,         
                "question": raw_train_data[idx]["question"],
                "metadata": {         
                    "stage": stage,                      
                    "original_idx": idx
                }
            }
            
            
            torch.save(data_pair, f"{output_dir}/s{stage}_idx{idx}.pt")

if __name__ == "__main__":
    extract_and_save()