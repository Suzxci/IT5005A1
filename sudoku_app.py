"""Complete Streamlit interface for the IT5005 Sudoku assignment.

The six assessed knowledge-base and solver functions stay in sudoku_solver.py.
This module only contains input validation, visual presentation, state handling,
and presentation of the solver's instrumented backward-chaining proof trace.
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    pl_bc_entails,
    pl_bc_entails_with_trace,
    solve_full_grid_bc,
    solve_full_grid_fc,
)


DATA_PATH = Path(__file__).with_name("puzzles.json")
TRACE_LIMIT = 80


@st.cache_data(show_spinner=False)
def load_pool(path: str) -> dict[str, Any]:
    """Load public puzzle inputs while deliberately excluding solutions."""
    with open(path, encoding="utf-8") as stream:
        raw = json.load(stream)

    n = int(raw["n"])
    box_h = int(raw["box_h"])
    box_w = int(raw["box_w"])
    if n <= 0 or box_h <= 0 or box_w <= 0 or box_h * box_w != n:
        raise ValueError("Invalid board or box dimensions in puzzles.json.")

    puzzles = []
    for index, item in enumerate(raw.get("puzzles", []), start=1):
        givens: dict[tuple[int, int], int] = {}
        for key, value in item.get("givens", {}).items():
            try:
                row, column = (int(part) for part in key.split("_"))
                digit = int(value)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Puzzle {index} contains an invalid given: {key!r}."
                ) from exc
            if not (1 <= row <= n and 1 <= column <= n and 1 <= digit <= n):
                raise ValueError(
                    f"Puzzle {index} contains an out-of-range given: {key}={digit}."
                )
            cell = (row, column)
            if cell in givens:
                raise ValueError(f"Puzzle {index} repeats cell {cell}.")
            givens[cell] = digit

        expected_count = int(item.get("given_count", len(givens)))
        if expected_count != len(givens):
            raise ValueError(
                f"Puzzle {index} says it has {expected_count} givens, "
                f"but {len(givens)} were found."
            )

        # Never copy item['solution']; app inference receives only givens.
        puzzles.append({"givens": givens, "given_count": len(givens)})

    if not puzzles:
        raise ValueError("puzzles.json does not contain any puzzles.")
    return {"n": n, "box_h": box_h, "box_w": box_w, "puzzles": puzzles}


@st.cache_resource(show_spinner=False)
def get_definite_kb(
    n: int,
    box_h: int,
    box_w: int,
    givens_items: tuple[tuple[tuple[int, int], int], ...],
):
    """Build one read-only definite KB per puzzle for query reuse."""
    return build_definite_kb(n, box_h, box_w, dict(givens_items))


def normalise_grid(raw_grid: Any, n: int) -> dict[tuple[int, int], int]:
    """Normalise the required mapping result, tolerating a list-grid result."""
    grid: dict[tuple[int, int], int] = {}
    if isinstance(raw_grid, Mapping):
        for key, value in raw_grid.items():
            if isinstance(key, tuple) and len(key) == 2:
                row, column = key
            elif isinstance(key, str) and "_" in key:
                row, column = key.split("_", maxsplit=1)
            else:
                raise ValueError(f"The solver returned an invalid cell key: {key!r}.")
            try:
                grid[(int(row), int(column))] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"The solver returned an invalid value for cell {key!r}."
                ) from exc
        return grid

    if isinstance(raw_grid, Sequence) and not isinstance(raw_grid, (str, bytes)):
        if len(raw_grid) != n:
            raise ValueError(f"The solver returned {len(raw_grid)} rows instead of {n}.")
        for row_index, row in enumerate(raw_grid, start=1):
            if not isinstance(row, Sequence) or len(row) != n:
                raise ValueError(f"Solver row {row_index} does not contain {n} cells.")
            for column_index, value in enumerate(row, start=1):
                grid[(row_index, column_index)] = int(value)
        return grid

    raise ValueError("The solver did not return a cell mapping or a 2D grid.")


def validate_solution(
    grid: Mapping[tuple[int, int], int],
    givens: Mapping[tuple[int, int], int],
    n: int,
    box_h: int,
    box_w: int,
) -> None:
    """Reject incomplete or invalid solver output before displaying it."""
    required = {(row, column) for row in range(1, n + 1) for column in range(1, n + 1)}
    missing = required - set(grid)
    extra = set(grid) - required
    if missing:
        raise ValueError(f"The solver returned an incomplete grid ({len(missing)} missing).")
    if extra:
        raise ValueError(f"The solver returned {len(extra)} out-of-range cells.")

    expected = set(range(1, n + 1))
    for cell in required:
        if grid[cell] not in expected:
            raise ValueError(f"Cell {cell} has an out-of-range value: {grid[cell]!r}.")
    for cell, value in givens.items():
        if grid[cell] != value:
            raise ValueError(f"The result changed the given value at cell {cell}.")

    for row in range(1, n + 1):
        if {grid[(row, column)] for column in range(1, n + 1)} != expected:
            raise ValueError(f"Row {row} violates the Sudoku constraints.")
    for column in range(1, n + 1):
        if {grid[(row, column)] for row in range(1, n + 1)} != expected:
            raise ValueError(f"Column {column} violates the Sudoku constraints.")
    for top in range(1, n + 1, box_h):
        for left in range(1, n + 1, box_w):
            values = {
                grid[(row, column)]
                for row in range(top, top + box_h)
                for column in range(left, left + box_w)
            }
            if values != expected:
                raise ValueError(f"The box beginning at ({top}, {left}) is invalid.")


def board_markup(
    values: Mapping[tuple[int, int], int],
    givens: Mapping[tuple[int, int], int],
    n: int,
    box_h: int,
    box_w: int,
) -> str:
    """Create an accessible board with visible box and cell-state styling."""
    rows = []
    for row in range(1, n + 1):
        cells = []
        for column in range(1, n + 1):
            value = values.get((row, column), "")
            is_given = (row, column) in givens
            cell_class = "given" if is_given else ("inferred" if value != "" else "empty")
            label = f"Row {row}, column {column}: {value if value != '' else 'empty'}"
            top = "3px" if row == 1 or (row - 1) % box_h == 0 else "1px"
            left = "3px" if column == 1 or (column - 1) % box_w == 0 else "1px"
            right = "3px" if column == n else "1px"
            bottom = "3px" if row == n else "1px"
            text = html.escape(str(value)) if value != "" else "&nbsp;"
            cells.append(
                f'<td class="{cell_class}" aria-label="{html.escape(label)}" '
                f'style="border-width:{top} {right} {bottom} {left};">{text}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <style>
      .sudoku-wrap {{ max-width: 520px; margin: .5rem auto 1rem; }}
      .sudoku-board {{ width: 100%; table-layout: fixed; border-collapse: collapse; }}
      .sudoku-board td {{ aspect-ratio: 1/1; border-style: solid; border-color: #344054;
        text-align: center; vertical-align: middle; font-size: clamp(1rem, 3vw, 1.55rem);
        line-height: 1; padding: 0; }}
      .sudoku-board td.given {{ background: #e8eef9; color: #101828; font-weight: 750; }}
      .sudoku-board td.inferred {{ background: #f7fbff; color: #175cd3; font-weight: 600; }}
      .sudoku-board td.empty {{ background: #fff; color: transparent; }}
      .board-legend {{ display:flex; justify-content:center; gap:1.25rem; color:#475467;
        font-size:.86rem; margin-bottom:.4rem; }}
      .legend-swatch {{ display:inline-block; width:.85rem; height:.85rem; margin-right:.35rem;
        border:1px solid #98a2b3; vertical-align:-.08rem; }}
      .legend-given {{ background:#e8eef9; }} .legend-inferred {{ background:#f7fbff; }}
    </style>
    <div class="board-legend">
      <span><span class="legend-swatch legend-given"></span>Given</span>
      <span><span class="legend-swatch legend-inferred"></span>Inferred</span>
    </div>
    <div class="sudoku-wrap"><table class="sudoku-board" aria-label="Sudoku board">
      <tbody>{''.join(rows)}</tbody></table></div>
    """


def visible_trace_steps(steps: list[Any]) -> list[tuple[int | None, Any | None]]:
    """Keep the proof readable while ensuring its final deduction is visible."""
    if len(steps) <= TRACE_LIMIT:
        return [(index, step) for index, step in enumerate(steps, start=1)]
    head_count = TRACE_LIMIT * 2 // 3
    tail_count = TRACE_LIMIT - head_count
    head = [(index, steps[index - 1]) for index in range(1, head_count + 1)]
    tail_start = len(steps) - tail_count + 1
    tail = [(index, steps[index - 1]) for index in range(tail_start, len(steps) + 1)]
    return head + [(None, None)] + tail


RESULT_KEYS = (
    "solved_grid",
    "solve_elapsed",
    "solved_with",
    "solve_error",
    "query_signature",
    "query_verdict",
    "query_elapsed",
    "query_trace",
    "query_error",
)


def clear_results() -> None:
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def reset_if_puzzle_changed(puzzle_index: int) -> None:
    if st.session_state.get("active_puzzle_index") != puzzle_index:
        clear_results()
        st.session_state["active_puzzle_index"] = puzzle_index


def main() -> None:
    st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="centered")
    st.title("Sudoku Solver")
    st.caption("Propositional-logic inference with forward and backward chaining")

    try:
        pool = load_pool(str(DATA_PATH))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Unable to load puzzles.json: {exc}")
        st.stop()

    n, box_h, box_w = pool["n"], pool["box_h"], pool["box_w"]
    puzzles = pool["puzzles"]

    st.subheader("1. Select a puzzle")
    puzzle_index = st.selectbox(
        "Puzzle",
        options=range(len(puzzles)),
        format_func=lambda index: (
            f"Puzzle {index + 1} - {puzzles[index]['given_count']} given cells"
        ),
        key="puzzle_selector",
    )
    reset_if_puzzle_changed(puzzle_index)
    givens = puzzles[puzzle_index]["givens"]
    st.markdown(board_markup(givens, givens, n, box_h, box_w), unsafe_allow_html=True)

    st.divider()
    st.subheader("2. Solve the full grid")
    algorithm = st.radio(
        "Inference algorithm",
        options=("Forward chaining", "Backward chaining"),
        horizontal=True,
        key="solver_algorithm",
    )
    if st.button("Solve puzzle", type="primary", use_container_width=True, key="solve_button"):
        solver = solve_full_grid_fc if algorithm == "Forward chaining" else solve_full_grid_bc
        for key in ("solved_grid", "solve_elapsed", "solved_with", "solve_error"):
            st.session_state.pop(key, None)
        try:
            with st.spinner(f"Solving with {algorithm.lower()}..."):
                started = time.perf_counter()
                raw_solution = solver(n, box_h, box_w, dict(givens))
                elapsed = time.perf_counter() - started
            solution = normalise_grid(raw_solution, n)
            validate_solution(solution, givens, n, box_h, box_w)
            st.session_state["solved_grid"] = solution
            st.session_state["solve_elapsed"] = elapsed
            st.session_state["solved_with"] = algorithm
        except Exception as exc:
            st.session_state["solve_error"] = f"The solver could not complete this puzzle: {exc}"

    if error := st.session_state.get("solve_error"):
        st.error(error)
    if solution := st.session_state.get("solved_grid"):
        st.success("Puzzle solved successfully.")
        algorithm_metric, time_metric = st.columns(2)
        algorithm_metric.metric("Algorithm", st.session_state["solved_with"])
        time_metric.metric("Elapsed time", f"{st.session_state['solve_elapsed']:.6f} s")
        st.markdown(board_markup(solution, givens, n, box_h, box_w), unsafe_allow_html=True)

    st.divider()
    st.subheader("3. Ask about one cell")
    st.write("Check whether the definite knowledge base entails a proposed value.")
    row_col, column_col, value_col = st.columns(3)
    row = int(row_col.number_input("Row", 1, n, 1, 1, key="query_row"))
    column = int(column_col.number_input("Column", 1, n, 1, 1, key="query_column"))
    value = int(value_col.number_input("Value", 1, n, 1, 1, key="query_value"))
    signature = (puzzle_index, row, column, value)
    if st.session_state.get("query_signature") not in (None, signature):
        for key in ("query_signature", "query_verdict", "query_elapsed", "query_trace", "query_error"):
            st.session_state.pop(key, None)

    if st.button("Check entailment", use_container_width=True, key="query_button"):
        for key in ("query_verdict", "query_elapsed", "query_trace", "query_error"):
            st.session_state.pop(key, None)
        try:
            kb = get_definite_kb(n, box_h, box_w, tuple(sorted(givens.items())))
            query = atom("Is", row, column, value)
            with st.spinner("Following the inference rules..."):
                started = time.perf_counter()
                verdict = bool(pl_bc_entails(kb, query))
                query_elapsed = time.perf_counter() - started
                trace_verdict, trace = pl_bc_entails_with_trace(kb, query)
                if trace_verdict != verdict:
                    raise RuntimeError(
                        "The backward-chaining verdict and its trace disagree."
                    )
            st.session_state["query_signature"] = signature
            st.session_state["query_verdict"] = verdict
            st.session_state["query_elapsed"] = query_elapsed
            st.session_state["query_trace"] = trace
        except Exception as exc:
            st.session_state["query_signature"] = signature
            st.session_state["query_error"] = f"The query could not be completed: {exc}"

    if st.session_state.get("query_signature") == signature:
        if error := st.session_state.get("query_error"):
            st.error(error)
        elif "query_verdict" in st.session_state:
            verdict = st.session_state["query_verdict"]
            message = f"Entailed: {verdict}"
            (st.success if verdict else st.info)(message)
            st.caption(
                f"Backward-chaining query time: {st.session_state['query_elapsed']:.6f} s"
            )

    st.divider()
    st.subheader("4. Reasoning trace (Tutor mode)")
    trace = (
        st.session_state.get("query_trace")
        if st.session_state.get("query_signature") == signature
        else None
    )
    if trace is None:
        st.info("Run the cell query above to generate a reasoning trace.")
        return

    st.caption(
        "These are the actual fact and rule dependencies used by the "
        "backward-chaining query, shown in premise-before-conclusion order."
    )
    verdict = st.session_state["query_verdict"]
    (st.success if verdict else st.info)(
        "A fact-grounded proof was found." if verdict
        else "The requested value was not proved; an explicit elimination proof is shown when available."
    )
    st.metric("Proof steps", len(trace))

    if not trace:
        st.write("No proof chain is available for this proposition.")
        return
    show_all_steps = False
    if len(trace) > TRACE_LIMIT:
        show_all_steps = st.checkbox(
            "Show every proof step",
            value=False,
            key="show_all_trace_steps",
        )
    if len(trace) > TRACE_LIMIT and not show_all_steps:
        st.info(
            f"This proof contains {len(trace)} steps. The first and final "
            f"parts are represented by {TRACE_LIMIT} displayed steps below. "
            "Select 'Show every proof step' to view the full trace."
        )

    displayed_steps = (
        list(enumerate(trace, start=1))
        if show_all_steps else visible_trace_steps(trace)
    )
    for step_number, step in displayed_steps:
        if step is None:
            st.markdown("**... intermediate proof steps omitted from the display ...**")
            continue
        with st.expander(
            f"Step {step_number}: {step['title']}",
            expanded=step_number <= 3,
        ):
            st.write(step["explanation"])


if __name__ == "__main__":
    main()
