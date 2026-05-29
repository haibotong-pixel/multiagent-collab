from macollab.scoring.numeric import NumericScorer
from macollab.tasks.reasoning_gsm8k import Gsm8kMiniSuite


def test_suite_has_name_and_tasks():
    suite = Gsm8kMiniSuite()
    assert suite.name == "gsm8k_mini"
    tasks = suite.tasks()
    assert len(tasks) >= 8
    assert all(t.id and t.prompt and t.type == "reasoning" for t in tasks)


def test_ground_truths_are_self_consistent():
    # Sanity: every ground truth is numeric and a perfect answer scores correct.
    scorer = NumericScorer()
    for t in Gsm8kMiniSuite().tasks():
        assert scorer.score(f"The answer is {t.ground_truth}.", t.ground_truth).correct
