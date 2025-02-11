# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional

from torch.distributed.fsdp import ShardingStrategy  # type: ignore
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # type: ignore
import torch


@dataclass
class train_config:
    model_name: str = "kotomamba"
    tokenizer_name: str = "kotomamba"

    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    fsdp_cpu_offload: bool = False
    fsdp_activation_checkpointing: bool = False

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = (
        StateDictType.SHARDED_STATE_DICT
    )

    num_epochs: int = 1
    train_iteration: int = 1

    batch_size: int = 4
    gradient_accumulation_steps: int = 1

    optimizer: str = "AdamW"
    lr: float = 1e-4
    lr_min: float = 1e-5
    lr_decay: float = 0.80  # ratio of decay
    lr_warmup: float = 0.002  # ratio of warmup
    lr_decay_style: str = "cosine"

    use_sequence_length_schedule: bool = False
    sequence_length_warmup_min: int = 8
    sequence_length_warmup: float = 0.15

    clip_grad_norm: float = 1.0
    weight_decay: float = 0.1
    gamma: float = 0.85
    adamw_eps: float = 1e-5
    adamw_betas: tuple[float, float] = (0.9, 0.95)

    seed: int = 42

    use_fp16: bool = False
    use_bf16: bool = False
    use_fp32: bool = False
    mixed_precision: bool = False
    param_dtype: Optional[torch.dtype] = None

    dataset: str = ""
    num_workers_dataloader: int = 1

    peft_method: str = "None"  # None , llama_adapter, prefix
    use_peft: bool = False

    output_dir: str = ""

    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False

    one_gpu: bool = False

    save_model: bool = False
    save_checkpoint_path: str = ""
    save_optimizer: bool = False  # will be used if using FSDP
    load_checkpoint_path: str = ""
    save_interval_iteration: int = 100
    no_load_rng: bool = False

    use_fast_kernels: bool = False
    use_mpi: bool = False

    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None

    val_iteration: int = 100
    run_validation: bool = False

    from_scratch: bool = False  # only for mamba
