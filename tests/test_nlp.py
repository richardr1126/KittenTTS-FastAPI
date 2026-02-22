from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import nlp
from nlp import TextPreprocessor


def test_markdown_pipe_tables_are_removed():
    preprocessor = TextPreprocessor()
    text = (
        "The results are below.\n"
        "| Column A | Column B |\n"
        "| --- | --- |\n"
        "| 10 | 20 |\n"
        "This concludes the experiment."
    )

    cleaned, metadata = preprocessor.process_with_metadata(text)

    assert "the results are below" in cleaned
    assert "this concludes the experiment" in cleaned
    assert "column a" not in cleaned
    assert metadata["table_lines_removed"] == 3


def test_inline_citations_and_structural_refs_are_removed():
    preprocessor = TextPreprocessor()
    text = "Prior work [12] and [3, 8] showed gains (Fig. 2) in this domain."

    cleaned, metadata = preprocessor.process_with_metadata(text)

    assert "prior work" in cleaned
    assert "showed gains" in cleaned
    assert "twelve" not in cleaned
    assert "fig" not in cleaned
    assert metadata["reference_fragments_removed"] >= 2


def test_symbol_noise_is_collapsed():
    preprocessor = TextPreprocessor()
    text = "Noise ____ |||| ==== ---- ##?? should not drown valid text."

    cleaned, metadata = preprocessor.process_with_metadata(text)

    assert "valid text" in cleaned
    assert "_" not in cleaned
    assert "|" not in cleaned
    assert metadata["symbol_noise_collapsed"] >= 1


def test_normal_prose_regression_behavior():
    preprocessor = TextPreprocessor()
    text = "Hello there, this is a normal paragraph about science and progress."
    cleaned = preprocessor(text)

    assert cleaned == "hello there this is a normal paragraph about science and progress"


def test_numbers_currency_and_time_still_normalize():
    preprocessor = TextPreprocessor()
    text = "It costs $12.50 at 3:05pm and rose 50%."
    cleaned = preprocessor(text)

    assert "twelve dollars and fifty cents" in cleaned
    assert "three oh five pm" in cleaned
    assert "fifty percent" in cleaned


def test_chunk_text_by_sentences_respects_chunk_size():
    text = "First sentence. Second sentence. Third sentence."
    chunks = nlp.chunk_text_by_sentences(text, chunk_size=20)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert len(chunks) >= 2


def test_chunk_text_by_sentences_handles_empty_input():
    assert nlp.chunk_text_by_sentences("", chunk_size=120) == []
    assert nlp.chunk_text_by_sentences("   ", chunk_size=120) == []
