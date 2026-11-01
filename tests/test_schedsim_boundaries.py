import pytest
from kv_serve_lab.schedsim import Request, SimConfig, Simulator


def test_late_arrival_metrics_are_relative_to_arrival():
    result = Simulator(SimConfig(64, 4, .01)).run([Request('late', 8, 2)], [.05])
    assert result.requests['late'].ttft_sec == pytest.approx(.01)
    assert result.requests['late'].latency_sec == pytest.approx(.02)


def test_idle_gap_does_not_trigger_token_based_timeout():
    result = Simulator(SimConfig(64, 4, .01)).run([Request('late', 1, 1)], [100.0])
    assert result.requests['late'].ttft_sec == pytest.approx(.01)


def test_decode_precedes_unfinished_prefill():
    cfg = SimConfig(4, 3, .01)
    result = Simulator(cfg).run([Request('long', 20, 2), Request('short', 1, 3)], [0, 0])
    # FCFS long prompt consumes first; this checks budget never exceeds 4.
    assert all(t <= 4 for t in result.step_input_tokens)
    assert result.total_tokens_computed == 24


def test_cost_model_and_littles_law():
    cfg = SimConfig(64, 4, step_overhead=.01, t_prefill_token=.001, t_decode_token=.002)
    result = Simulator(cfg).run([Request('a', 10, 3), Request('b', 10, 3)], [0, 0])
    # first forward .01 + 20*.001 = .03; decode steps .014 each
    assert result.requests['a'].ttft_sec == pytest.approx(.03)
    assert result.requests['a'].latency_sec == pytest.approx(.058)
    assert result.mean_system_requests == pytest.approx(result.arrival_rate * result.metrics().mean_latency_sec)
    assert result.output_throughput == pytest.approx(6/.058)


@pytest.mark.parametrize('field', ['step_sec', 'step_overhead', 't_prefill_token', 't_decode_token'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_costs_rejected(field, value):
    with pytest.raises(ValueError):
        SimConfig(64, 4, **{**{'step_sec': .01}, field: value})


def test_cost_model_rejects_zero_duration_decode_steps():
    with pytest.raises(ValueError):
        SimConfig(64, 4, step_sec=.01, t_prefill_token=.001)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_invalid_arrival(value):
    with pytest.raises(ValueError):
        Simulator(SimConfig(64, 4, .01)).run([Request('a', 1, 1)], [value])


def test_population_integral_is_measured_during_steps():
    # [0,.01]: a 在系统中；[.01,.02]: a 全程、b 在 .015 到达；[.02,.03]: b。
    res = Simulator(SimConfig(1, 1, .01)).run(
        [Request('a', 1, 2), Request('b', 1, 1)], [0, .015])
    assert res.step_system_areas == pytest.approx([.01, .015, .01])
    assert res.mean_system_requests == pytest.approx(.035 / .03)
    assert res.mean_system_requests == pytest.approx(
        res.arrival_rate * res.metrics().mean_latency_sec)


def test_saturated_throughput_includes_prefill_only_steps():
    res = Simulator(SimConfig(1, 1, step_sec=.01)).run([Request('r', 100, 1)], [0.0])
    assert res.steady_state_throughput == pytest.approx(1.0)


class TestIntegerLimits:
    """token 数与并发上限必须是整数：小数会让统计口径失真。"""

    @pytest.mark.parametrize('field', ['max_num_batched_tokens', 'max_num_seqs'])
    @pytest.mark.parametrize('value', [float('nan'), 1.5, '8'])
    def test_config_rejects_non_integer_limits(self, field, value):
        with pytest.raises(ValueError, match=field):
            SimConfig(**{**{'max_num_batched_tokens': 64, 'max_num_seqs': 4,
                            'step_sec': .01}, field: value})

    @pytest.mark.parametrize('prompt,output', [(2.5, 2), (2, 2.5), (True, 1)])
    def test_request_rejects_non_integer_token_counts(self, prompt, output):
        with pytest.raises(ValueError):
            Request('a', prompt_tokens=prompt, output_tokens=output)
