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


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_invalid_arrival(value):
    with pytest.raises(ValueError):
        Simulator(SimConfig(64, 4, .01)).run([Request('a', 1, 1)], [value])
