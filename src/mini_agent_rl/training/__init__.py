"""训练数据边界；真正的 SFT/GRPO 权重更新将在下一阶段接入。"""

from .data import export_sft_messages
from .synthetic import generate_dataset, generate_split, validate_record
from .hotpot import download_hotpot_subset, prepare_final_test
from .audit import audit_sft_directory, audit_final_test, audit_rl_signal
from .sft import encode_assistant_only, AssistantOnlyCollator, train_sft
from .logprob import compare_logprobs, stable_same_instance
from .grpo import GRPOConfig, RewardConfig, MinimalGRPOTrainer, clipped_grpo_loss, rollout_equal_grpo_loss
