"""CPU assertion that installed Transformers accepts a Trainer with no eval set."""
from transformers import TrainingArguments
from mini_agent_rl.training.sft import _evaluation_arguments

options=_evaluation_arguments(False,50)
args=TrainingArguments(output_dir="/tmp/v133-goal-j-no-eval",report_to=[],**options)
assert str(args.eval_strategy).lower().endswith("no")
assert args.eval_steps is None
print({"eval_strategy":str(args.eval_strategy),"eval_steps":args.eval_steps,"eval_dataset":None})
