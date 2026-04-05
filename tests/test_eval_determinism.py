from src.callbacks.eval_callback import EvaluationSchedule


def test_evaluation_schedule_uses_step_zero_and_checkpoint_cadence() -> None:
    schedule = EvaluationSchedule()
    assert schedule.deterministic is True
    assert schedule.should_evaluate(0) is True
    assert schedule.should_evaluate(10_000) is True
    assert schedule.should_evaluate(15_000) is False
