# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import logging

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states

logger = logging.getLogger(__name__)

try:
    from integration.megatron_schedule_runtime import (
        dmi_abort_cuda_graph_capture,
        dmi_begin_cuda_graph_capture,
        dmi_current_phase,
        dmi_finish_cuda_graph_capture,
        dmi_finish_full_iteration_replay,
        dmi_force_eager_unit,
        dmi_prepare_full_iteration_replay,
    )
except Exception:
    def dmi_begin_cuda_graph_capture(
        *,
        warmup_enabled=True,
        full_iteration=False,
        valid_counts_by_microbatch=None,
    ):
        del warmup_enabled, full_iteration, valid_counts_by_microbatch

    def dmi_finish_cuda_graph_capture():
        return None

    def dmi_current_phase(default="validation"):
        return str(default)

    def dmi_abort_cuda_graph_capture():
        pass

    def dmi_prepare_full_iteration_replay(plan, valid_counts_by_microbatch):
        del plan, valid_counts_by_microbatch
        return False

    def dmi_finish_full_iteration_replay():
        pass

    class dmi_force_eager_unit:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

# The below functions traverse through nested data structures (tuples, lists, dicts)
# present in src and creates a deep copy where all PyTorch tensors are cloned,
# detached from the computation graph, and moved to CUDA device. Non-tensor objects
# are returned as-is.


def copy_tensors_in_struct(src):
    """Copy src to new tensors."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src):
    """Copy src to pre-existing tensors in tgt."""
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


# Class to copy dataloader output to static CUDA tensors for CUDA graph input. This
# maintains separate static buffers for training and validation CUDA graphs.
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(copy_tensors_in_struct(inputs))
        else:

            for k in inputs.keys():
                if k not in StaticBufferLoader.static_buffers[stage][microbatch]:
                    if isinstance(inputs[k], torch.Tensor):
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = torch.empty_like(
                            inputs[k], device="cuda"
                        )
                    else:
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = inputs[k]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]

    @staticmethod
    def iter_static_buffers(stage, num_microbatches):
        assert stage in ['training', 'validation']
        return iter(StaticBufferLoader.static_buffers[stage][:num_microbatches])


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    dmi_plans = {'training': None, 'validation': None}

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps

    @staticmethod
    def _extract_dmi_valid_count(inputs):
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]
        if not isinstance(inputs, dict):
            return None
        valid_count = inputs.get('dmi_valid_count', None)
        if valid_count is None:
            return None
        if isinstance(valid_count, torch.Tensor):
            return [int(x) for x in valid_count.detach().cpu().view(-1).tolist()]
        if isinstance(valid_count, (list, tuple)):
            return [int(x) for x in valid_count]
        return [int(valid_count)]

    @staticmethod
    def _record_valid_count(valid_counts_by_microbatch, microbatch, valid_count):
        if valid_count is None:
            return
        previous = valid_counts_by_microbatch[microbatch]
        if previous is None:
            valid_counts_by_microbatch[microbatch] = list(valid_count)
            return
        if list(previous) != list(valid_count):
            raise RuntimeError(
                "DMI full-iteration valid_count mismatch for microbatch "
                f"{microbatch}: {previous} != {valid_count}"
            )

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        valid_counts_by_microbatch = [None for _ in range(num_microbatches)]
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    inputs = next(iterator0)
                    self._record_valid_count(
                        valid_counts_by_microbatch,
                        b,
                        self._extract_dmi_valid_count(inputs),
                    )
                    data_list.append(self.static_loader(inputs, 'training' if training else 'validation', b))
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        inputs = next(data_iterator[i])
                        self._record_valid_count(
                            valid_counts_by_microbatch,
                            b,
                            self._extract_dmi_valid_count(inputs),
                        )
                        data_list_i.append(self.static_loader(inputs, 'training' if training else 'validation', b))
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_list, valid_counts_by_microbatch

    def static_data_list(self, data_iterator, model, training, num_microbatches):
        """Rebuild data iterators over static buffers without re-reading the dataloader."""
        stage = 'training' if training else 'validation'
        if not isinstance(model, list) or len(model) == 1:
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            return [self.static_loader.iter_static_buffers(stage, num_microbatches)] if iterator0 is not None else [None]

        assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
        return [
            self.static_loader.iter_static_buffers(stage, num_microbatches)
            if data_iterator[i] is not None
            else None
            for i in range(len(model))
        ]

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        data_list, valid_counts_by_microbatch = self.data_read(
            data_iterator, model, training, num_microbatches
        )
        kwargs['data_iterator'] = data_list

        training_str = 'training' if training else dmi_current_phase(default='validation')
        if training_str not in FullCudaGraphWrapper.curr_iteration:
            FullCudaGraphWrapper.curr_iteration[training_str] = 0
            FullCudaGraphWrapper.cuda_graph[training_str] = None
            FullCudaGraphWrapper.result[training_str] = None
            FullCudaGraphWrapper.dmi_plans[training_str] = None
        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            logger.info(f'Capture CUDA graph for {training_str}!!!')
            torch.distributed.barrier()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = torch.cuda.Stream()
            dmi_begin_cuda_graph_capture(
                warmup_enabled=True,
                full_iteration=True,
                valid_counts_by_microbatch=valid_counts_by_microbatch,
            )
            try:
                with torch.cuda.graph(
                    FullCudaGraphWrapper.cuda_graph[training_str],
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                        *args, **kwargs
                    )
                dmi_plan = dmi_finish_cuda_graph_capture()
                if dmi_plan is not None:
                    FullCudaGraphWrapper.dmi_plans[training_str] = dmi_plan
            except Exception:
                dmi_abort_cuda_graph_capture()
                raise
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'CUDA graph capture done for {training_str}!!!')

        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            dmi_plan = FullCudaGraphWrapper.dmi_plans.get(training_str, None)
            fallback_to_eager = dmi_prepare_full_iteration_replay(
                dmi_plan,
                valid_counts_by_microbatch,
            ) if dmi_plan is not None else False
            if fallback_to_eager:
                kwargs['data_iterator'] = self.static_data_list(
                    data_iterator,
                    model,
                    training,
                    num_microbatches,
                )
                with dmi_force_eager_unit():
                    FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                        *args, **kwargs
                    )
            else:
                FullCudaGraphWrapper.cuda_graph[training_str].replay()
                dmi_finish_full_iteration_replay()

        self.next_iter(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1
