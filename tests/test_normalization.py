from src.plate_ocr_pipeline import (
    normalize_plate_text,
    digit_match_cost,
    letter_match_cost,
    source_plate_hints,
)


def test_arabic_digits_are_normalized():
    assert "1234" in normalize_plate_text("ا ب ح ١٢٣٤")


def test_attached_letters_and_digits_are_separated():
    value = normalize_plate_text("ابح1234")
    assert "1234" in value


def test_similar_digits_cost_less_than_unrelated_digits():
    assert digit_match_cost("1238", "1233") < digit_match_cost("1238", "1290")


def test_exact_letters_cost_zero():
    assert letter_match_cost("ابح", "ابح") == 0.0


def test_hint_generation():
    hints = source_plate_hints("ا ب ح 1234")
    assert hints
