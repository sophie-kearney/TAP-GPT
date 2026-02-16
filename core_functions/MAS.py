###
# DATA ABSTRACTION
###

import pandas as pd

###
# DEFINE AGENT WRAPPER
###

class Agent:
    def __init__(self, name, model, tokenizer, adapter_name, system_prompt):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def call(self, user_message, max_new_tokens=256):
        self.messages.append({"role": "user", "content": user_message})

        self.model.set_adapter(self.adapter_name)

        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        self.messages.append({"role": "assistant", "content": response})
        return response


def load_modality_agent(task, k, p, seed, modality, base_model, tokenizer):
    # load data
    file_name = f"data/task_prompts/{task}_{seed}_{modality}_{k}_MAS.csv"
    prompts = pd.read_csv(file_name)

    # process data
    prompts = prompts.rename(columns={
        'prompt': 'input_text',
        'AlzheimersDisease': 'output_text'
    })
    test_data = prompts[prompts['split'] == 'test'].drop(columns=['split'])

    # adjust prompt to ask for some interpretability
    instruct_enhanced = (
        "Below is patient record. Each column contains features related to Alzheimer's disease.\n"

        "Predict whether the patient in the last row has Alzheimer's disease (1) or does not (0). "
        "In your reasoning, provide up to 3 brief evidence statements grounded in this modality only "
        "and refer to several specific brain regions (by their column names) that most strongly influence the prediction.\n\n"

        "Respond ONLY with a single JSON object with keys: "
        f"prediction (0 or 1), confidence (low/medium/high), reasoning (string). \n"
        "Do not include any text before or after the JSON."
    )
    test_data['input_text'] = test_data['input_text'].str.replace(r'(?<=Instruction: )(.*?)(?=\n)',
                                                                  instruct_enhanced,
                                                                  regex=True)
    test_data['input_text'] = test_data['input_text'].str.replace(
        r'(?<=Response:)(.*?)(?=\n|$)',
        " Let's think step by step.",
        regex=True
    )

    system_prompts = {
        "amyloid": "You are an amyloid PET expert for Alzheimer's disease. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "tau": "You are a tau PET expert for Alzheimer's disease. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "MRI": "You are an MRI neurodegeneration expert. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "frontal": "You are an expert in amyloid PET, tau PET, and MRI volume measurements in the frontal lobe. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "temporal": "You are an expert in amyloid PET, tau PET, and MRI volume measurements in the temporal lobe. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "parietal": "You are an expert in amyloid PET, tau PET, and MRI volume measurements in the parietal lobe. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "occipital": "You are an expert in amyloid PET, tau PET, and MRI volume measurements in the occipital lobe. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
        "subcortical": "You are an expert in amyloid PET, tau PET, and MRI volume measurements in the subcortical region. Answer ONLY with a single JSON object with keys: prediction (int 0 or 1), probability (float 0-1), reasoning (string). No prose before or after.",
    }

    agent = Agent(
        name=modality,
        model=base_model,
        tokenizer=tokenizer,
        adapter_name=modality,
        system_prompt=system_prompts[modality]
    )

    return test_data, agent