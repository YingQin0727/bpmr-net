from bpmr_net.demo import run_demo


def test_demo_runs():
    result = run_demo(seed=1701, size=24)
    assert result["feature_shape"][1] == 400
    assert result["bpmr_trainable_parameters"] == 383
    assert result["b0_probability_sum_max_error"] < 1e-5
    assert result["bpmr_probability_sum_max_error"] < 1e-5
