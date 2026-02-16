###
# DATA ABSTRACTION
###
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, sys
import pandas as pd
from transformers import LogitsProcessorList
pd.set_option('display.max_columns', None)

###
# ONLY ALLOW 1 OR 0 CLASS
###

# creates mask for all tokens besides the ones we specifcy
class OnlyAllowSpecificTokensProcessor(torch.nn.Module):
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed_token_ids = allowed_token_ids

    def forward(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask

###
# VANILLA MODELS FROM HUGGINGFACE
###

def load_huggingface_model(model_path_name, model_name, all_prompts):
    # --- LOAD MODEL ----
    tokenizer = AutoTokenizer.from_pretrained(model_path_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # --- ONLY ALLOW 0 or 1 ---
    # gets the token IDs for the values we want
    # allowed_ids = tokenizer.convert_tokens_to_ids(['0', '1'])
    ids0 = tokenizer.encode("0", add_special_tokens=False)
    ids1 = tokenizer.encode("1", add_special_tokens=False)
    ids0s = tokenizer.encode(" 0", add_special_tokens=False)
    ids1s = tokenizer.encode(" 1", add_special_tokens=False)
    allowed_ids = sorted(set(
        ([ids0[0]] if len(ids0) == 1 else []) +
        ([ids1[0]] if len(ids1) == 1 else []) +
        ([ids0s[0]] if len(ids0s) == 1 else []) +
        ([ids1s[0]] if len(ids1s) == 1 else [])
    ))
    if len(allowed_ids) < 2:
        raise ValueError(f"Could not find distinct single-token ids for 0/1. Found: {allowed_ids}")

    # restrict model output to only 0 and 1
    logits_processor = LogitsProcessorList([
        OnlyAllowSpecificTokensProcessor(allowed_ids)
    ])

    # --- GENERATE PREDICTIONS ---
    results = []

    tasks = all_prompts['task'].unique()
    for t in tasks:
        curr_task = all_prompts[all_prompts['task'] == t]

        batch_size = 16

        for i in range(0, len(curr_task), batch_size):
            batch = curr_task.iloc[i:i + batch_size]
            prompts = batch['prompt'].tolist()

            messages_batch = [
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}]
                for prompt in prompts
            ]

            # apply chat template and tokenize in batch
            if model_name=="Qwen3":
                text_batch = [tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True,
                                                            enable_thinking=False) for
                              messages in messages_batch]
            else:
                text_batch = [tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True) for
                              messages in messages_batch]
            model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # generate predictions
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1,
                do_sample=False,
                logits_processor=logits_processor
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # store results in format we can make into pandas df
            for j, (_, row) in enumerate(batch.iterrows()):
                results.append({
                    'task': t,
                    'model': model_name,
                    'ID': row['ID'],
                    'AlzheimersDisease': row['AlzheimersDisease'],
                    'prompt': row['prompt'],
                    'prediction': predictions[j].strip()
                })

    return  pd.DataFrame(results)

def load_openai_model(model_path_name, model_name, DATABRICKS_TOKEN, API_URL, all_prompts):
    # --- LOAD MODEL ---
    client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=API_URL
    )

    # --- GENERATE RESULTS ---
    results = []

    tasks = all_prompts['task'].unique()
    for t in tasks:
        curr_task = all_prompts[all_prompts['task'] == t]

        for _, row in curr_task.iterrows():
            response = client.chat.completions.create(
                model=model_path_name,
                messages=[
                    {
                        "role": "user",
                        "content": row['prompt']
                    }
                ]
            )
            results.append({
                'task': t,
                'model': model_name,
                'ID': row['ID'],
                'AlzheimersDisease': row['AlzheimersDisease'],
                'prompt': row['prompt'],
                'prediction': response.choices[0].message.content.strip(),
            })

    return pd.DataFrame(results)