import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

def get_model_request(prompt):
    print("--- final model prompt ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## model preparation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)

    history = [
        {"role": "system", "content": prompt}
    ]

    history.append({"role":"user", "content": "answer in json format"})

    inputs = tokenizer.apply_chat_template(
        history,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = 'pt'
    ).to(device)

    outputs = model.generate(
            **inputs, 
            max_new_tokens = 256,
            do_sample = True,
            temperature = 1.0,
            top_p = 0.9,
            repetition_penalty = 1.03,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    return response



