from macollab.scoring.numeric import NumericScorer, extract_final_number


def test_extract_final_number_takes_last_number():
    assert extract_final_number("First 12, then the answer is 42") == 42.0
    assert extract_final_number("total = 1,234") == 1234.0
    assert extract_final_number("the answer is -3.5") == -3.5
    assert extract_final_number("no digits here") is None


def test_numeric_scorer_correct_and_incorrect():
    s = NumericScorer()
    good = s.score("After working it out, the answer is 42.", 42)
    bad = s.score("I think it's 41.", 42)
    assert good.correct is True and good.score == 1.0
    assert bad.correct is False and bad.score == 0.0


def test_numeric_scorer_no_number_is_incorrect():
    s = NumericScorer()
    assert s.score("I am not sure.", 42).correct is False
