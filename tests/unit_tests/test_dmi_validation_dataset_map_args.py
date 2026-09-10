# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse

from megatron.training.arguments import _add_dmi_args


def test_validation_dataset_map_dir_argument():
    parser = _add_dmi_args(argparse.ArgumentParser())

    assert parser.parse_args([]).dmi_validation_dataset_map_dir is None
    assert (
        parser.parse_args(
            ["--dmi-validation-dataset-map-dir", "/tmp/validation-maps"]
        ).dmi_validation_dataset_map_dir
        == "/tmp/validation-maps"
    )
