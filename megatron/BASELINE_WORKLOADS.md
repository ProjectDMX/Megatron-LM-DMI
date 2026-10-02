# Baseline capture workloads

Enable capture using `BASELINE_HIDDEN_METRICS_DIR` and select a workload with
`BASELINE_EVAL_WORKLOAD`. PyTorch hooks and NNsight sync use
`BASELINE_CAPTURE_MODE=immediate`; the native NNsight async branch uses
`BASELINE_CAPTURE_MODE=cache_async`.

| Workload | Captures |
|---|---|
| `router_logits` | Raw router logits |
| `qk_weights` | Rank-deduplicated compute-model Q/K fragments |
| `ep_full` | Full weighted expert outputs, inverse maps, selected expert IDs |
| `ep_sampled` | Configurably sampled weighted outputs; full maps and IDs |
| `training_health_full` | `ep_full`, Q/K weights, routing weights |
| `training_health_sampled` | `ep_sampled`, Q/K weights, routing weights |
| `validation_quality` | Pre-final-RMSNorm residuals, router logits, sample loss means/counts; validation only |
| `final_hidden_states` | Pre-final-RMSNorm residuals |

Historical `hidden_states`, `ep_outputs`, `multi_signal`, and `vocab_raw` remain
available. EP workloads require the existing dropless, unfused all-to-all path.
The external baselines run eagerly and retain Transformer Engine computation.
Q/K capture excludes linear-attention projections and gated-attention gate rows.
It runs before optimizer mutation, including skipped updates; initial/resume
snapshots are outside timed training iterations. Payloads use packed uint8 byte
fragments with original dtype/shape/placement in `weight_layouts.json` and events.

## Configurable sampling

Set `BASELINE_HOOK_CONFIG` to a YAML file:

```yaml
hooks:
  moe_packed_weighted_output:
    source_sampling:
      function: megatron.baseline_sampling.round_robin
      args: {count: 2, offset: 0}
      iteration_field: global_batch_id
      iteration_origin: 1
```

The MIND built-in function path is accepted as an alias without importing DMI.
Custom dotted callables accept `iteration`, `num_sources`, and configured keyword
arguments. They must return a deterministic, fixed nonzero number of distinct
in-range source positions. Sources are ordered by expert-TP rank, then EP rank,
within each expert-DP group. Omitting sampling retains full capture. Unsupported
hooks warn and ignore the sampling configuration. The sampled workloads require
an explicit selector; full workloads reject one to prevent mislabeled results.
The policy and selected positions are recorded with source counts and expert
identity for reconstruction. Empty destination outputs are retained.

## Validation and output

`validation_quality` uses the configured validation schedule; training forwards
are unobserved. Each validation invocation has a pass ID and each validation
batch has its own expected microbatch count. `BASELINE_VALIDATION_METRICS_DIR`
records the same native, synchronized evaluation-pass timer in the original
unmonitored branch and the capture branches (`validation_passes.jsonl`).
Per-batch records label host-call/capture-completion timing separately; use
whole-pass durations for monitored versus unmonitored validation comparisons. Capture timing includes native CPU
completion; correctness checks and final metadata serialization follow it.

Tests under `tests/unit_tests/baselines/` cover selectors, actual routing outputs,
EP row selection, Q/K layout and deduplication, native capture, and phase gating.
