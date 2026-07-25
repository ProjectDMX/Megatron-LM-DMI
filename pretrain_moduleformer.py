#!/usr/bin/env python3
"""SFT entry point for the released MoLM ModuleFormer architecture."""

from __future__ import annotations

import os
import time

from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    inprocess_restart,
    pretrain,
    set_startup_timestamps,
)
from megatron.training.arguments import core_transformer_config_from_args

from dmi_training_tool.sft_mixture.moduleformer_pipeline import (
    ModuleFormerConfig,
    ModuleFormerPipelineModel,
)
from integration.megatron_hook_requirements import parse_hook_selection
from pretrain_gpt import (
    forward_step,
    get_embedding_ranks,
    train_valid_test_datasets_provider,
)


_PROGRAM_START_TIME = time.time()


def add_moduleformer_args(parser):
    group = parser.add_argument_group(title="ModuleFormer")
    group.add_argument(
        "--moduleformer-config",
        type=str,
        required=True,
        help="Local ModuleFormer config.json or directory containing it.",
    )
    group.add_argument(
        "--moduleformer-hf-checkpoint",
        type=str,
        default=None,
        help="Local official HF checkpoint used when --load is not supplied.",
    )
    group.add_argument(
        "--moduleformer-train-routers",
        action="store_true",
        help="Train router parameters. The paper-aligned default freezes them.",
    )
    group.add_argument(
        "--moduleformer-force-deterministic-expert-combine",
        action="store_true",
        help=(
            "Use fixed-order expert combination for determinism tests. "
            "The default preserves the released ModuleFormer index_add path."
        ),
    )
    return parser


def _dmi_enabled(args) -> bool:
    return bool(
        getattr(args, "dmi_enable", None)
        or str(os.getenv("DMI_ENABLE", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    )


def _validate_model_contract(args, hf_config: ModuleFormerConfig) -> None:
    expected = {
        "hidden_size": int(hf_config.n_embd),
        "num_layers": int(hf_config.n_layer),
        "num_attention_heads": int(hf_config.n_head),
        "ffn_hidden_size": int(hf_config.ffd_hidden),
        "padded_vocab_size": int(hf_config.vocab_size),
    }
    mismatches = {
        name: (int(getattr(args, name)), expected_value)
        for name, expected_value in expected.items()
        if int(getattr(args, name)) != expected_value
    }
    if mismatches:
        raise ValueError(
            "Megatron arguments do not match the released ModuleFormer config: "
            f"{mismatches}"
        )
    if not bool(args.sft):
        raise ValueError("ModuleFormer training entry point requires --sft")
    if int(args.tensor_model_parallel_size) != 1:
        raise NotImplementedError("ModuleFormer first runnable path requires TP=1")
    if int(args.context_parallel_size) != 1:
        raise NotImplementedError("ModuleFormer first runnable path requires CP=1")
    if int(args.expert_model_parallel_size) != 1:
        raise NotImplementedError("ModuleFormer first runnable path requires EP=1")
    if int(args.pipeline_model_parallel_size) != 2:
        raise ValueError("ModuleFormer local verification requires PP=2")
    if int(args.micro_batch_size) != 1:
        raise ValueError("ModuleFormer local verification requires MBS=1")
    if not bool(args.untie_embeddings_and_output_weights):
        raise ValueError("Released ModuleFormer requires untied embeddings/output weights")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        raise NotImplementedError("ModuleFormer first runnable path does not use virtual PP")
    if float(hf_config.aux_loss_weight) != 0.0:
        raise ValueError("Expected the released MoLM aux_loss_weight=0 contract")

    has_megatron_checkpoint = bool(getattr(args, "load", None))
    has_hf_checkpoint = bool(args.moduleformer_hf_checkpoint)
    if has_megatron_checkpoint == has_hf_checkpoint:
        raise ValueError(
            "Specify exactly one initial checkpoint source: --load or "
            "--moduleformer-hf-checkpoint"
        )


def moduleformer_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    del pg_collection
    if vp_stage is not None:
        raise NotImplementedError("ModuleFormer does not support virtual pipeline stages")
    args = get_args()
    hf_config = ModuleFormerConfig.from_pretrained(args.moduleformer_config)
    hf_config.force_deterministic_expert_combine = bool(
        args.moduleformer_force_deterministic_expert_combine
    )
    _validate_model_contract(args, hf_config)
    transformer_config = (
        core_transformer_config_from_args(args) if config is None else config
    )

    selected_hooks: set[str] = set()
    if _dmi_enabled(args):
        selected_hooks = parse_hook_selection(
            getattr(args, "dmi_hook_selection", None)
            or os.getenv("DMI_HOOK_SELECTION", "router-summary")
        )
    segment_capacity = None
    if "router-summary" in selected_hooks:
        row_capacity = getattr(args, "dmi_packed_max_conversations_per_row", None)
        if row_capacity is None or int(row_capacity) <= 0:
            raise ValueError(
                "ModuleFormer DMI router-summary requires "
                "--dmi-packed-max-conversations-per-row"
            )
        segment_capacity = int(args.micro_batch_size) * int(row_capacity)

    return ModuleFormerPipelineModel(
        transformer_config,
        hf_config,
        pre_process=pre_process,
        post_process=post_process,
        hf_checkpoint_path=(
            None if getattr(args, "load", None) else args.moduleformer_hf_checkpoint
        ),
        install_dmi_router_summary="router-summary" in selected_hooks,
        segment_capacity=segment_capacity,
        freeze_routers=not bool(args.moduleformer_train_routers),
    )


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(
        program_start=_PROGRAM_START_TIME,
        main_entry=_MAIN_ENTRY_TIME,
    )

    train_valid_test_datasets_provider.is_distributed = True
    train_valid_test_datasets_provider.dmi_standard_dataset_provider = True
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(
        pretrain
    )
    wrapped_pretrain(
        train_valid_test_datasets_provider,
        moduleformer_model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_moduleformer_args,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
