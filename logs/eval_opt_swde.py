import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from lm_eval import evaluator, tasks

# Параметры
model_name = "facebook/opt-125m"
task_list = ["swde"]
num_fewshot = 0
batch_size = 2

print(f"Evaluating {model_name} on {task_list}")

# Запуск оценки
results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_name},dtype=float16",
    tasks=task_list,
    num_fewshot=num_fewshot,
    batch_size=batch_size,
    device="cuda",
    log_samples=True
)

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
for task, result in results["results"].items():
    print(f"\nTask: {task}")
    for metric, value in result.items():
        print(f"  {metric}: {value}")
