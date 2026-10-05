from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from megatron.baseline_evaluation import setup_hidden_state_evaluation
from megatron.baseline_validation import ValidationBoundary
import megatron.baseline_workload_evaluation as workloads


def args(**kw):
    a=dict(skip_train=True,use_legacy_models=False,perform_rl_step=False,cuda_graph_impl='none',
           virtual_pipeline_model_parallel_size=None,context_parallel_size=1,bf16=True,
           recompute_granularity=None,overlap_moe_expert_parallel_comm=False,eval_iters=2,do_valid=True)
    a.update(kw)
    return SimpleNamespace(**a)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for k in ('BASELINE_HIDDEN_METRICS_DIR','BASELINE_EVAL_WORKLOAD','BASELINE_CAPTURE_MODE','BASELINE_VALIDATION_METRICS_DIR'):
        monkeypatch.delenv(k,raising=False)


@pytest.mark.parametrize('skip_train',[True,False])
def test_validation_capture_constructed(monkeypatch,tmp_path,skip_train):
    monkeypatch.setenv('BASELINE_HIDDEN_METRICS_DIR',str(tmp_path))
    monkeypatch.setenv('BASELINE_EVAL_WORKLOAD','validation_quality')
    constructor=Mock();monkeypatch.setattr(workloads,'WorkloadEvaluation',constructor)
    model=SimpleNamespace();a=args(skip_train=skip_train)
    assert setup_hidden_state_evaluation([model],a) is constructor.return_value
    constructor.assert_called_once_with(model,a,str(tmp_path),'immediate','validation_quality')


def test_requested_capture_without_output_rejected(monkeypatch):
    monkeypatch.setenv('BASELINE_EVAL_WORKLOAD','validation_quality')
    with pytest.raises(ValueError,match='METRICS_DIR'):setup_hidden_state_evaluation([SimpleNamespace()],args())


@pytest.mark.parametrize('change',[{'eval_iters':0},{'do_valid':False}])
def test_validation_workload_without_validation_rejected(monkeypatch,tmp_path,change):
    monkeypatch.setenv('BASELINE_HIDDEN_METRICS_DIR',str(tmp_path))
    monkeypatch.setenv('BASELINE_EVAL_WORKLOAD','validation_quality')
    with pytest.raises(ValueError,match='enabled validation'):setup_hidden_state_evaluation([SimpleNamespace()],args(**change))


def test_skip_training_workload_rejected(monkeypatch,tmp_path):
    monkeypatch.setenv('BASELINE_HIDDEN_METRICS_DIR',str(tmp_path))
    with pytest.raises(ValueError,match='validation_quality'):setup_hidden_state_evaluation([SimpleNamespace()],args())


@pytest.mark.parametrize('phase',['validation','test'])
def test_missing_capture_cannot_silently_time_evaluation(monkeypatch,phase):
    monkeypatch.setenv('BASELINE_EVAL_WORKLOAD','validation_quality')
    with pytest.raises(RuntimeError,match='not initialized'):ValidationBoundary([SimpleNamespace()],phase=phase)


def test_unmonitored_evaluation_still_works():
    b=ValidationBoundary([SimpleNamespace()]);b.begin(1,4);b.end(1)
    assert len(b.rows)==1 and b.rows[0]['duration_ns']>0


def test_initialized_evaluation_drives_capture(monkeypatch):
    monkeypatch.setenv('BASELINE_EVAL_WORKLOAD','validation_quality')
    capture=Mock();b=ValidationBoundary([SimpleNamespace(_baseline_hidden_evaluation=capture)])
    b.begin(1,4);b.end(1)
    capture.begin_iteration.assert_called_once_with(0,phase='validation',microbatches=4,pass_id=1)
    capture.end_iteration.assert_called_once_with(0)
