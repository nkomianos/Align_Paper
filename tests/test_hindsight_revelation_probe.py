from interaction_sprint.hindsight_revelation_probe import cases, teacher_text, history_text


def test_balanced_orders_and_no_hidden_preference_in_base():
    rows=cases()
    assert len(rows)==16
    for i in range(0,16,2):
        assert rows[i]['options']==rows[i+1]['options'][::-1]
        assert 'actual preference' not in rows[i]['prompt']
        assert rows[i]['domain']==rows[i+1]['domain']


def test_previous_action_separate_from_feedback():
    c=cases()[0]
    text=teacher_text(c,0,'My actual preference is coffee.')
    assert text.startswith(c['prompt'])
    assert 'previously answered: A.' in text
    assert 'next user message was: My actual preference is coffee.' in text


def test_history_control_preserves_feedback_and_replays_original():
    c=cases()[0]
    feedback='My actual preference is coffee. Please use that preference.'
    assert history_text(c,1,'shown')==teacher_text(c,0,feedback)
    shown=history_text(c,1,'shown')
    assert history_text(c,1,'omitted')==shown.replace('The assistant previously answered: A.\n','')
    assert history_text(c,1,'redacted')==shown.replace('answered: A.','answered: [redacted].')
