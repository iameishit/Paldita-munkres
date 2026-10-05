"""Step-by-step traces."""

import munkres
from munkres import solve

INF = float("inf")


# --------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------


def test_trace_records_every_kind_of_step_and_ends_with_the_answer():
    result = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]], trace=True)
    kinds = {e["event"] for e in result.trace.events}
    assert kinds == {"row_reduce", "star", "augment_start", "adjust", "augmented", "done"}
    assert result.trace.events[-1] == {"event": "done", "pairs": [(0, 1), (1, 0), (2, 2)]}
    assert len(result.trace) == len(result.trace.events) == len(result.trace.steps())
    assert solve([[1]]).trace is None


def test_trace_text_and_html_use_plain_language():
    text = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]], trace=True).trace.to_text()
    assert text.startswith("Hungarian algorithm on a 3x3 matrix (10 steps)")
    assert "Flip the path through column 0: row 1 is now matched." in text
    page = solve([[4, 1, 3], [2, 0, 5], [3, 2, 2]], trace=True).trace.to_html()
    assert page.startswith("<!doctype html>") and "<script" not in page and "<ol>" in page


def test_trace_of_tall_matrix_swaps_the_words():
    trace = solve([[1, 2], [3, 4], [0, 9]], trace=True).trace
    assert trace.transposed is True
    assert all("column" in s or "Done" in s for s in trace.steps()[:1])
    assert trace.to_text().count("Reduce column") == 2  # the solver's rows are your columns


def test_trace_html_escapes_markup():
    trace = munkres.Trace(shape=(1, 1), events=[{"event": "done", "pairs": ["<b>"]}])
    assert "&lt;b&gt;" in trace.to_html() and "<b>" not in trace.to_html()


def test_trace_with_gating_and_with_nothing_to_match():
    assert solve([[1, 9], [9, 1]], max_cost=3, trace=True).trace.events[-1]["event"] == "done"
    nothing = solve([[1, 2]], max_cost=-INF, trace=True)
    assert nothing.trace.events == [{"event": "done", "pairs": []}]
