"""Matched forward-only validation timing for capture and unmonitored runs."""
import json
import os
from pathlib import Path
from time import perf_counter_ns


class ValidationBoundary:
    def __init__(self, model, *, phase="validation"):
        self.phase = phase
        self.evaluation = getattr(model[0], '_baseline_hidden_evaluation', None)
        self.output = os.environ.get('BASELINE_VALIDATION_METRICS_DIR')
        self.pass_id = getattr(model[0], '_baseline_validation_pass_id', 0) + 1
        model[0]._baseline_validation_pass_id = self.pass_id
        self.rows = []

    def begin(self, batch, microbatches):
        self.start = perf_counter_ns()
        if self.evaluation is not None:
            self.evaluation.begin_iteration(batch-1, phase=self.phase,
                                            microbatches=microbatches, pass_id=self.pass_id)

    def end(self, batch):
        if self.evaluation is not None:
            self.evaluation.end_iteration(0)
        end = perf_counter_ns()
        self.rows.append(dict(phase=self.phase, pass_id=self.pass_id, iteration=batch, duration_ns=end-self.start, timing_scope='host forward call plus capture completion'))

    def close(self, *, elapsed_seconds=None):
        if self.output:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            folder = Path(self.output) / f'rank_{rank:05d}'
            folder.mkdir(parents=True, exist_ok=True)
            # Native evaluate timer already synchronizes; do not add another wait.
            with (folder / 'validation_passes.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(phase=self.phase, pass_id=self.pass_id,
                    batches=len(self.rows), duration_ns=(None if elapsed_seconds is None
                    else round(elapsed_seconds * 1e9)), complete=elapsed_seconds is not None,
                    timing_scope='native synchronized evaluate timer'))+'\n')
            with (folder / 'validation_iterations.jsonl').open('a') as stream:
                for row in self.rows:
                    stream.write(json.dumps(row)+'\n')
