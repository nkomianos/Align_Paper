from audit_wrench_fenced_transport import transport


def test_exact_outer_fence_only():
    assert transport('```json\n{"score":0.3}\n```')==(.3,'fenced_json')
    assert transport('{"score":0.3}')==(.3,'bare_json')
    for text in ('Reasoning\n```json\n{"score":0.3}\n```','```json\n{"score":true}\n```',
                 '```json\n{"score":0.3,"other":1}\n```'):
        assert transport(text)[0] is None
