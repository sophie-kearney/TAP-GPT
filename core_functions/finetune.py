###
# DATA ABSTRACTION
###

import os, json
import copy
import pandas as pd
from typing import Dict, Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import LogitsProcessorList
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import sys, os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    Seq2SeqTrainer
)
import random, numpy as np
from core_functions.models import OnlyAllowSpecificTokensProcessor
from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

###
# SET UP CLASSES
###

IGNORE_INDEX = -100

class DataCollatorForCausalLM(object):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        source_max_len: int,
        target_max_len: int,
        train_on_source: bool = False, # Labels are constructed by directly copying the target sequences.
        predict_with_generate: bool = False
    ):
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input_text']}" for example in instances]
        targets = [f"{example['output_text']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

###
# FINETUNE HUGGINGFACE MODEL
###

def finetune_huggingface(READ_TOKEN, WRITE_TOKEN, train_data, test_data,
                         new_model_name, base_model_name, seed):
    random.seed(seed)
    np.random.seed(seed)

    if "tau" in new_model_name:
        lora_r = 32
        lora_dropout = 0.1
        learning_rate = 0.000190
        batch_size = 8
        weight_decay = 0.001
        max_steps = 90
        lr_scheduler_type = 'cosine'
    elif "MRI" in new_model_name:
        lora_r = 32
        lora_dropout = 0.1
        learning_rate = 0.000190
        batch_size = 8
        weight_decay = 0.001
        max_steps = 90
        lr_scheduler_type = 'cosine'
    elif "amyloid" in new_model_name:
        lora_r = 8
        lora_dropout = 0.1
        learning_rate = 5e-5
        batch_size = 16
        weight_decay = 0.01
        max_steps = -1
        lr_scheduler_type = 'linear'

    print(new_model_name)
    print("lora_r", lora_r, "lora_dropout", lora_dropout, "learning_rate", learning_rate,
          "batch_size", batch_size, "weight_decay", weight_decay, "max_steps", max_steps,
          "lr_scheduler_type", lr_scheduler_type)

    # --- LOAD BASE MODEL ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        token=READ_TOKEN
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="right",
        use_fast=False,
        token=READ_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # --- PREP LORA ---

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules="all-linear",  # The names of the modules to apply the adapter to
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    # --- PROCESS DATA ---

    source_max_len = 2048
    target_max_len = 1

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=source_max_len,
        target_max_len=target_max_len,
    )

    data_module = dict(
        train_dataset=train_data,
        eval_dataset=test_data,
        predict_dataset=None,
        data_collator=data_collator
    )

    # --- DEFINE TRAINING ARGUMENTS ---

    training_args = Seq2SeqTrainingArguments(
        output_dir=f'models/{new_model_name}',
        learning_rate=learning_rate,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=1,  # sy11
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=batch_size,  # 16, # 8,  # sy11
        weight_decay=weight_decay,  # 0.01,   #0.0,  #sy11
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        max_grad_norm=0.3,
        num_train_epochs=3.0,
        max_steps=max_steps,  # 200, #50,  # overides num_train_epochs  # sy11
        lr_scheduler_type=lr_scheduler_type,  # 'linear',   # 'constant',  # sy11
        warmup_ratio=0.03,
        warmup_steps=0,
        logging_strategy='steps',
        logging_steps=10,
        save_strategy='steps',
        save_steps=25,
        save_total_limit=4,
        save_safetensors=True,
        seed=0,
        data_seed=42,
        fp16=True,
        fp16_opt_level='O1',
        half_precision_backend='auto',
        eval_steps=25,
        dataloader_num_workers=3,
        optim='paged_adamw_32bit',
        group_by_length=False,
        dataloader_pin_memory=True,
        push_to_hub=True,
        # TODO - set huggingface user
        hub_model_id=f"<user>/{new_model_name}",
        hub_strategy='every_save',
        hub_token=WRITE_TOKEN,
        hub_private_repo=True,
        gradient_checkpointing=True,
        fp16_backend='auto',
        remove_unused_columns=False,
        generation_config=transformers.GenerationConfig(max_new_tokens=1, do_sample=False,
                                                        pad_token_id=tokenizer.eos_token_id)
    )

    # --- FINETUNE MODEL ---
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )
    model.config.use_cache = False

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    return train_result, eval_result

###
# EVAL FINETUNED MODEL
###

def eval_finetuned(model, tokenizer, test_data, system_prompt="You are a helpful assistant."):
    # gets the token IDs for the values we want
    allowed_ids = tokenizer.convert_tokens_to_ids(['0', '1'])
    # restrict model output to only 0 and 1
    logits_processor = LogitsProcessorList([
        OnlyAllowSpecificTokensProcessor(allowed_ids)
    ])

    batch_size = 4
    results = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data.iloc[i:i + batch_size]
        prompts = batch['input_text'].tolist()

        messages_batch = [
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        # apply chat template and tokenize in batch
        text_batch = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for
                      messages in messages_batch]
        model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # generate predictions
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
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
                'model': "TAP-GPT",
                'ID': row['ID'],
                'AlzheimersDisease': row['output_text'],
                'prompt': row['input_text'],
                'prediction': predictions[j].strip()
            })
    return pd.DataFrame(results)

def eval_interpreatable(model, tokenizer, test_data, system_prompt="You are a helpful assistant."):
    batch_size = 6
    results = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data.iloc[i:i + batch_size]
        prompts = batch['input_text'].tolist()

        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]

        # apply chat template and tokenize in batch
        text_batch = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for
                      messages in messages_batch]
        model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # generate predictions
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                use_cache=True,
                return_dict_in_generate=False,
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
                'model': "Interpretable TAP-GPT",
                'ID': row['ID'],
                'AlzheimersDisease': row['output_text'],
                'prompt': row['input_text'],
                'raw_response': raw,
                'prediction': pred_label,
                'probability': prob,
                'reasoning': reasoning,
            })
    return pd.DataFrame(results)

###
# TEST IF EXISTS
###

def model_exists(model_name):
    try:
        model_info(model_name)
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError:
        return False