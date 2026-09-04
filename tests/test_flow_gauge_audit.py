import numpy as np
import pytest
import runpy

from interaction_sprint.flow_gauge_audit import audit, flow, matrix, variance


def test_exact_same_marginal_and_endpoint():
    points = np.array([[1., 0.], [0., 1.], [2., 3.]])
    for eps in (0., .01, .1):
        assert np.allclose(flow(0., points, eps), points)
        assert np.allclose(flow(1., points, eps), points)
        for t in (.1, .5, .9):
            a = matrix(t, eps)
            assert np.allclose(variance(t)*(a+a.T), (4*t-2)*np.eye(2))


def test_null_positive_control_and_derivative():
    result = audit()
    assert len(result['rows']) == 24
    for r in result['rows']:
        assert r['endpoint_paired_max_error'] < 1e-12
        assert r['continuity_residual'] < 1e-12
        assert r['analytic_flow_derivative_error'] < 1e-8
        if r['epsilon'] == 0:
            assert r['null_VFD'] == 0 and r['FM_excess_loss'] == 0
        if r['steps'] == 10000:
            assert abs(r['genuine_mean_shift_VFD']-r['genuine_mean_shift_KL']) < 1e-8
    assert any(r['ranks_null_above_shift'] for r in result['rows'])


def test_real_runner_manifest_and_tamper_detection(tmp_path, monkeypatch):
    from scripts.verify_flow_gauge_audit import verify
    root = tmp_path/'audit'
    monkeypatch.setattr('sys.argv', ['flow_gauge_audit', '--root', str(root)])
    runpy.run_module('interaction_sprint.flow_gauge_audit', run_name='__main__')
    assert verify(root)['grid_cells'] == 24
    with (root/'RESULT.json').open('a', encoding='utf-8') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(root)
