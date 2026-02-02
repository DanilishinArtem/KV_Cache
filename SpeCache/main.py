from lm_eval import evaluator, tasks as task_registry
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from specache import patch_attention

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model.cuda()
    print(model)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        max_length=getattr(tokenizer, "model_max_length", 4096),
        batch_size=1,
        trust_remote_code=True
    )
    return lm

def main():
    specache_config = {"bits": 8, "topK": 10}
    patch_attention(specache_config=specache_config)

    task = "swde"
    # task = "arc_easy"
    name_of_model = "Qwen/Qwen2-1.5B"
    
    model = load_model(name_of_model)

    tm = task_registry.TaskManager()
    results = evaluator.simple_evaluate(model=model,tasks=[task],task_manager=tm)
    table = make_table(results)
    print(table)


if __name__ == "__main__":
    main()