#!/usr/bin/env python3
"""Convert the official ModuleFormer checkpoint into a Megatron checkpoint."""

from __future__ import annotations

import torch

from megatron.core.enums import ModelType
from megatron.training import get_args
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_model

from pretrain_moduleformer import (
    add_moduleformer_args,
    moduleformer_model_provider,
)


def main() -> int:
    initialize_megatron(extra_args_provider=add_moduleformer_args)
    args = get_args()
    if not args.save:
        raise ValueError("ModuleFormer conversion requires --save")
    if args.load:
        raise ValueError("HF-to-Megatron conversion must not use --load")
    if args.ckpt_format != "torch_dist":
        raise ValueError("ModuleFormer conversion requires --ckpt-format torch_dist")

    model = get_model(
        moduleformer_model_provider,
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    save_checkpoint(
        0,
        model,
        None,
        None,
        num_floating_point_operations_so_far=0,
    )
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"ModuleFormer Megatron checkpoint written to {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
