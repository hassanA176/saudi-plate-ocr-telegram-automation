from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

ARABIC_TO_ENGLISH_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

OCR_LETTER_FIX = str.maketrans({
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("آ"): "ا",
    ord("ٱ"): "ا",
    ord("ذ"): "د",
    ord("ز"): "ر",
    ord("ج"): "ح",
    ord("خ"): "ح",
    ord("ت"): "ب",
    ord("ث"): "ب",
    ord("ش"): "س",
    ord("ض"): "ص",
    ord("ظ"): "ط",
    ord("غ"): "ع",
    ord("ف"): "ق",
    ord("ة"): "ه",
    ord("ھ"): "ه",
    ord("ؤ"): "و",
    ord("ي"): "ى",
    ord("ئ"): "ى",
    ord("ک"): "ك",
    ord("ڪ"): "ك",
    ord("گ"): "ك",
    ord("پ"): "ب",
})

VALID_PLATE_CHARS = set("ابح درسصطعقكلمنهوى".replace(" ", ""))


@dataclass(slots=True)
class OCRText:
    text: str
    confidence: float = 0.0
    engine: str = ""


@dataclass(slots=True)
class PlateResult:
    found: bool
    plate: str = ""
    confidence: float = 0.0
    engine: str = ""
    source_texts: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    matched_from_reference: bool = False


def normalize_plate_text(text: str) -> str:
    value = str(text or "").strip()
    value = value.translate(ARABIC_TO_ENGLISH_DIGITS)
    value = value.translate(OCR_LETTER_FIX)
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("هـ", "ه").replace("ة", "ه")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"([^\d\s])(\d)", r"\1 \2", value)
    value = re.sub(r"(\d)([^\d\s])", r"\1 \2", value)
    return re.sub(r"\s+", " ", value).strip()


def image_is_too_blurry(image_path: str | Path, threshold: float | None = None) -> bool:
    """Reject obviously blurry inputs before running OCR."""
    threshold = threshold if threshold is not None else float(
        os.getenv("OCR_BLUR_THRESHOLD", "30")
    )
    try:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        return float(variance) < float(threshold)
    except Exception:
        return False


def letters_from_source(text: str) -> str:
    normalized = normalize_plate_text(text)
    return "".join(char for char in normalized if char in VALID_PLATE_CHARS)


def numbers_from_source(text: str) -> list[str]:
    normalized = normalize_plate_text(text)
    values: list[str] = []
    for group in re.findall(r"\d+", normalized):
        if len(group) >= 2 and group not in values:
            values.append(group)
        if len(group) > 4:
            for value in (group[:4], group[-4:]):
                if value not in values:
                    values.append(value)
    return values


def source_plate_hints(text: str) -> list[str]:
    """Build plate candidates from a single noisy OCR text region."""
    normalized = normalize_plate_text(text)
    letters = "".join(char for char in normalized if char in VALID_PLATE_CHARS)
    digit_groups = re.findall(r"\d+", normalized)
    if not letters or not digit_groups:
        return []

    letter_windows: list[str] = []
    for count in range(min(3, len(letters)), 0, -1):
        for start in range(0, len(letters) - count + 1):
            value = letters[start:start + count]
            if value and value not in letter_windows:
                letter_windows.append(value)

    number_options: list[str] = []
    for group in digit_groups:
        if 2 <= len(group) <= 4 and group not in number_options:
            number_options.append(group)
        if len(group) > 4:
            for value in (group[:4], group[-4:]):
                if value not in number_options:
                    number_options.append(value)

    for start in range(len(digit_groups)):
        for end in range(start + 2, min(len(digit_groups), start + 4) + 1):
            sequence = digit_groups[start:end]
            for value in ("".join(sequence), "".join(reversed(sequence))):
                if 2 <= len(value) <= 4 and value not in number_options:
                    number_options.append(value)

    hints: list[str] = []
    for letter_value in letter_windows[:8]:
        spaced_letters = " ".join(letter_value)
        for number in number_options[:12]:
            hints.append(f"{spaced_letters} {number}")
    return hints


def combined_plate_hints(source_texts: Sequence[str]) -> list[str]:
    """Combine OCR regions when letters and digits were detected separately."""
    letter_sources: list[str] = []
    number_sources: list[str] = []

    for text in source_texts:
        letters = letters_from_source(text)
        if 1 <= len(letters) <= 4 and letters not in letter_sources:
            letter_sources.append(letters)
        for number in numbers_from_source(text):
            if number not in number_sources:
                number_sources.append(number)

    letter_options: list[str] = []
    for letters in letter_sources:
        for count in range(min(3, len(letters)), 0, -1):
            for start in range(0, len(letters) - count + 1):
                value = letters[start:start + count]
                if value and value not in letter_options:
                    letter_options.append(value)

    for left in letter_sources:
        for right in letter_sources:
            if left == right:
                continue
            for value in (
                (left[:1] + right)[:3],
                (right[:1] + left)[:3],
                (left + right[-1:])[:3],
                (right + left[-1:])[:3],
            ):
                if len(value) == 3 and value not in letter_options:
                    letter_options.append(value)

    hints: list[str] = []
    for letters in letter_options[:16]:
        spaced_letters = " ".join(letters)
        for number in number_sources[:16]:
            hints.append(f"{spaced_letters} {number}")
    return hints


def plate_parts(plate_text: str) -> tuple[str, str]:
    normalized = normalize_plate_text(plate_text)
    letters = "".join(char for char in normalized if char in VALID_PLATE_CHARS)
    digit_groups = re.findall(r"\d+", normalized)
    numbers = max(digit_groups, key=len) if digit_groups else ""
    return letters, numbers


def digit_match_cost(source: str, target: str) -> float:
    if not source or not target:
        return 9.0
    if source == target:
        return 0.0

    similar_pairs = {
        frozenset(("4", "5")): 0.45,
        frozenset(("4", "2")): 0.65,
        frozenset(("3", "8")): 0.55,
        frozenset(("6", "8")): 0.55,
        frozenset(("0", "6")): 0.65,
        frozenset(("0", "9")): 0.65,
        frozenset(("1", "7")): 0.70,
    }

    def char_cost(left: str, right: str) -> float:
        if left == right:
            return 0.0
        return similar_pairs.get(frozenset((left, right)), 1.0)

    if len(source) == len(target):
        return sum(char_cost(left, right) for left, right in zip(source, target))

    if len(source) < len(target):
        best = 9.0
        for start in range(0, len(target) - len(source) + 1):
            segment = target[start:start + len(source)]
            segment_cost = sum(char_cost(a, b) for a, b in zip(source, segment))
            missing_penalty = (len(target) - len(source)) * 0.48
            edge_bonus = -0.18 if start + len(source) == len(target) else 0.0
            best = min(best, segment_cost + missing_penalty + edge_bonus)
        return max(0.0, best)

    return 2.0 + abs(len(source) - len(target)) * 0.85


def letter_match_cost(source: str, target: str) -> float:
    if not source or not target:
        return 9.0

    cost = abs(len(source) - len(target)) * 1.35
    similar_pairs = {
        frozenset(("د", "ر")): 0.55,
        frozenset(("ا", "ى")): 0.55,
        frozenset(("ا", "ل")): 0.65,
        frozenset(("ب", "ن")): 0.70,
        frozenset(("ب", "ى")): 0.75,
        frozenset(("س", "ص")): 0.65,
        frozenset(("م", "ص")): 0.65,
        frozenset(("م", "ن")): 0.70,
        frozenset(("ق", "ع")): 0.85,
        frozenset(("ك", "ل")): 0.65,
        frozenset(("ه", "و")): 0.75,
    }
    for left, right in zip(source, target):
        if left != right:
            cost += similar_pairs.get(frozenset((left, right)), 1.0)
    return cost


def best_reference_match(candidates: Sequence[str], reference_plates: Iterable[str]) -> str:
    """Return a reference match only when the score is sufficiently unambiguous."""
    rows = []
    for reference in reference_plates:
        normalized = normalize_plate_text(reference)
        letters, numbers = plate_parts(normalized)
        if normalized and letters and numbers:
            rows.append((normalized, letters, numbers))

    scored: list[tuple[float, str, str, float, float]] = []
    for rank, candidate in enumerate(candidates):
        source_letters, source_numbers = plate_parts(candidate)
        if not source_letters or len(source_numbers) < 2:
            continue
        if len(source_numbers) < 3 and len(source_letters) < 3:
            continue

        for target_plate, target_letters, target_numbers in rows:
            digit_cost = digit_match_cost(source_numbers, target_numbers)
            letter_cost = letter_match_cost(source_letters, target_letters)

            if digit_cost == 0:
                if letter_cost > 1.55:
                    continue
            elif digit_cost <= 1.25:
                if len(source_numbers) < 4 and letter_cost > 0.75:
                    continue
                if letter_cost > 1.20:
                    continue
            elif digit_cost <= 1.60:
                if letter_cost > 0.45:
                    continue
            else:
                continue

            total = (digit_cost * 2.4) + letter_cost + (rank * 0.04)
            scored.append(
                (total, target_plate, candidate, digit_cost, letter_cost)
            )

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0])
    best = scored[0]
    if len(scored) > 1 and scored[1][0] - best[0] < 0.18 and best[3] > 0:
        return ""
    return best[1]


class RapidOCREngine:
    """Thin adapter around RapidOCR. Imported lazily for faster CLI startup."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def _load(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def read(self, image_path: str | Path) -> list[OCRText]:
        engine = self._load()
        result, _ = engine(str(image_path))
        output: list[OCRText] = []
        for item in result or []:
            try:
                _, text, confidence = item
                output.append(OCRText(str(text), float(confidence), self.name))
            except (TypeError, ValueError):
                continue
        return output


class EasyOCREngine:
    """Fallback OCR adapter. Loads the model only if needed."""

    name = "easyocr"

    def __init__(self) -> None:
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["ar", "en"], gpu=False)
        return self._reader

    def read(self, image_path: str | Path) -> list[OCRText]:
        reader = self._load()
        output: list[OCRText] = []
        for _, text, confidence in reader.readtext(str(image_path)):
            output.append(OCRText(str(text), float(confidence), self.name))
        return output


class HybridPlateOCR:
    """
    Portfolio refactor of the original OCR orchestration:
      blur gate -> RapidOCR -> candidate reconstruction -> reference matching
      -> optional EasyOCR fallback.
    """

    def __init__(
        self,
        reference_plates: Iterable[str] = (),
        use_easyocr_fallback: bool = True,
    ) -> None:
        self.reference_plates = tuple(reference_plates)
        self.fast_engine = RapidOCREngine()
        self.deep_engine = EasyOCREngine() if use_easyocr_fallback else None

    def _candidate_list(self, reads: Sequence[OCRText]) -> list[str]:
        texts = [item.text for item in reads if item.text.strip()]
        candidates: list[str] = []

        for text in texts:
            normalized = normalize_plate_text(text)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
            for hint in source_plate_hints(text):
                if hint not in candidates:
                    candidates.append(hint)

        for hint in combined_plate_hints(texts):
            if hint not in candidates:
                candidates.append(hint)

        return candidates

    def _resolve(
        self,
        reads: Sequence[OCRText],
        engine_name: str,
    ) -> PlateResult:
        candidates = self._candidate_list(reads)
        reference_match = best_reference_match(candidates, self.reference_plates)
        if reference_match:
            confidence = max((r.confidence for r in reads), default=0.0)
            return PlateResult(
                found=True,
                plate=reference_match,
                confidence=confidence,
                engine=engine_name,
                source_texts=[r.text for r in reads],
                candidates=candidates,
                matched_from_reference=True,
            )

        for candidate in candidates:
            letters, numbers = plate_parts(candidate)
            if len(letters) >= 3 and 2 <= len(numbers) <= 4:
                confidence = max((r.confidence for r in reads), default=0.0)
                return PlateResult(
                    found=True,
                    plate=normalize_plate_text(candidate),
                    confidence=confidence,
                    engine=engine_name,
                    source_texts=[r.text for r in reads],
                    candidates=candidates,
                )

        return PlateResult(
            found=False,
            engine=engine_name,
            source_texts=[r.text for r in reads],
            candidates=candidates,
        )

    def recognize(self, image_path: str | Path) -> PlateResult:
        if image_is_too_blurry(image_path):
            return PlateResult(found=False, engine="quality-gate")

        fast_reads = self.fast_engine.read(image_path)
        fast_result = self._resolve(fast_reads, self.fast_engine.name)
        if fast_result.found and (
            fast_result.matched_from_reference or not self.deep_engine
        ):
            return fast_result

        if self.deep_engine and fast_result.candidates:
            deep_reads = self.deep_engine.read(image_path)
            merged_reads = [*fast_reads, *deep_reads]
            deep_result = self._resolve(merged_reads, self.deep_engine.name)
            if deep_result.found:
                return deep_result

        return fast_result
