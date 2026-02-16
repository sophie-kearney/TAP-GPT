###
# DATA ABSTRACTION
###

import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import os
import torch
import sys, random, numpy as np
import json


###
# LOAD DATASET
###

seed = int(sys.argv[1]) if len(sys.argv) > 2 else 36
modality = sys.argv[2] if len(sys.argv) > 3 else "amyloid"
k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
p = int(sys.argv[4]) if len(sys.argv) > 4 else 16
task = sys.argv[5] if len(sys.argv) > 5 else "few_shot_serialized"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# load data
file_name = f"data/task_prompts/{task}_{seed}_{modality}_{p}_{k}.csv"
prompts = pd.read_csv(file_name)
# process data
prompts = prompts.rename(columns={
    'prompt': 'input_text',
    'AlzheimersDisease': 'output_text'
})

test_data = prompts[prompts['split'] == 'test'].drop(columns=['split'])

instruct_enhanced = (
    "Below is a table of patient records. Each column contains features related to Alzheimer's disease. "
    "The last row is missing a value in the 'AlzheimersDisease' column. Based on the patterns in the other rows, "
    "predict whether the patient in the last row has Alzheimer's disease (1) or does not (0). In your reasoning, refer to several specific brain regions (by their column names) that most strongly influence the prediction."
    "Respond ONLY with a single JSON object with keys: "
    f"prediction (0 or 1), probability (0-1 float), reasoning (string). "
    "Do not include any text before or after the JSON."
)


test_data['input_text'] = test_data['input_text'].str.replace(r'(?<=Instruction: )(.*?)(?=\n)',
                                                              instruct_enhanced,
                                                              regex=True)

test_data['input_text'] = test_data['input_text'].str.replace(
    r'(?<=Response:)(.*?)(?=\n|$)',
    "Let's think step by step.",
    regex=True
)


###
# LOAD MODEL
###

READ_TOKEN = os.getenv("READ_TOKEN")
# TODO - change huggingface user
model_name = f"<user>/tableGPT2-{task}-{seed}-{modality}-{p}-{k}"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    use_fast=False
)

tokenizer.pad_token = tokenizer.eos_token

###
# GENERATE RESPONSE
###

batch_size = 16
results = []
out_path = f"data/task_results/TAP-GPT_{task}_{seed}_{modality}_interpretable.csv"

for i in range(0, len(test_data), batch_size):
    batch = test_data.iloc[i:i + batch_size]
    prompts = batch['input_text'].tolist()

    messages_batch = [
        [
            {"role": "system", "content": (
                "You are TAP-GPT, a tabular clinical assistant. Answer ONLY with a single JSON object with keys: "
                f"prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after."
            )},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]

    # apply chat template and tokenize in batch
    text_batch = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for
                  messages in messages_batch]
    model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # generate predictions
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for j, (_, row) in enumerate(batch.iterrows()):
        raw = predictions[j].strip()
        pred_label, prob, reasoning = None, None, ""
        try:
            obj = json.loads(raw)
            pred_label = int(obj.get("prediction")) if obj.get("prediction") is not None else None
            prob = float(obj.get("probability")) if obj.get("probability") is not None else None
            reasoning = str(obj.get("reasoning", ""))
        except Exception:
            import re
            m = re.search(r'([01])\b(?!.*[01])', raw)
            pred_label = int(m.group(1)) if m else None
            reasoning = raw

        results.append({
            'task': task,
            'model': "Interpretable TAP-GPT",
            'ID': row['ID'],
            'AlzheimersDisease': row['output_text'],
            'prompt': row['input_text'],
            'modality': modality,
            'raw_response': raw,
            'pred_label': pred_label,
            'probability': prob,
            'reasoning': reasoning,
            'seed': seed
        })
        pd.DataFrame(results).to_csv(out_path, index=False)

results_df = pd.DataFrame(results)
results_df.to_csv(out_path, index=False)