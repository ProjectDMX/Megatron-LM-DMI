# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from types import SimpleNamespace

import pretrain_gpt
from megatron.training.arguments import _add_data_args


def _dataset_args(*, keep_last_partial_validation_sequence):
    return SimpleNamespace(
        allow_ambiguous_pad_tokens=False,
        context_parallel_size=1,
        create_attention_mask_in_dataloader=True,
        data_cache_path=None,
        data_parallel_size=1,
        dataloader_defer_npy_index_mmap=False,
        dataloader_fast_cache_load=False,
        dmi_enable=False,
        eod_mask_loss=False,
        fim_data=False,
        full_validation=True,
        global_batch_size=1,
        hybrid_context_parallel=False,
        keep_last_partial_validation_sequence=keep_last_partial_validation_sequence,
        micro_batch_size=1,
        mid_level_dataset_surplus=0.005,
        mmap_bin_files=True,
        multiple_validation_sets=True,
        num_dataset_builder_threads=1,
        num_workers=0,
        object_storage_cache_path=None,
        per_dataset_sequences_path=None,
        reset_attention_mask=False,
        reset_position_ids=False,
        seed=1234,
        seq_length=16,
        sequence_parallel=False,
        sft=False,
        split="1,1,1",
        tensor_model_parallel_size=1,
        train_iters=1,
    )


def test_keep_last_partial_validation_sequence_argument():
    parser = _add_data_args(argparse.ArgumentParser())

    assert parser.parse_args([]).keep_last_partial_validation_sequence is False
    assert (
        parser.parse_args(
            ["--keep-last-partial-validation-sequence"]
        ).keep_last_partial_validation_sequence
        is True
    )


def test_keep_last_partial_validation_sequence_config_mapping(monkeypatch):
    tokenizer = SimpleNamespace(vocab_size=128)
    monkeypatch.setattr(pretrain_gpt, "build_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(
        pretrain_gpt,
        "get_blend_and_blend_per_split",
        lambda args: (None, None),
    )

    default_config = pretrain_gpt.core_gpt_dataset_config_from_args(
        _dataset_args(keep_last_partial_validation_sequence=False)
    )
    retained_config = pretrain_gpt.core_gpt_dataset_config_from_args(
        _dataset_args(keep_last_partial_validation_sequence=True)
    )

    assert default_config.drop_last_partial_validation_sequence is True
    assert retained_config.drop_last_partial_validation_sequence is False
