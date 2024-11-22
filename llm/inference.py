import os
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def chat(prompt: str, chat_history: List[Dict[str, str]]) -> str:
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.layers": "cuda:0",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-2b-it",
        device_map=device_map,
        torch_dtype=torch.float16,
        token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    )

    # peft_config = PeftConfig.from_pretrained("./recyclelm")

    model = PeftModel.from_pretrained(base_model, "./recyclelm")

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-1.1-2b-it", token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token

    def generate_text(prompt: str, max_length: int = 700) -> str:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                top_k=10,
                top_p=0.95,
                temperature=0.7,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    prompt_enhanced = """
    You are RecycleBot, an expert assistant dedicated to helping users recycle items effectively. You understand the rules and best practices for recycling in various regions and provide clear, actionable advice. Always offer tips that are environmentally friendly and practical. If you're unsure, suggest consulting local recycling guidelines.


    Chat History: {chat_history}

    User's Message: {user_message}

    Your Response:
    """
    result = generate_text(
        prompt_enhanced.format(chat_history=chat_history, user_message=prompt)
    )
    return result
