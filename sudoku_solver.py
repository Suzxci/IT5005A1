"""IT5005 Assignment 1: student implementation file.

Implement the functions marked below. Do not modify utils.py or logic_.py.
"""

from utils import *
from logic_ import *


# Do not change this function; it is used to create atomic propositions.
def atom(prefix, r, c, v):
    """prefix is 'Is' or 'Not'. Returns the Expr for e.g. Is3_2_4."""
    return expr(f'{prefix}{r}_{c}_{v}')


def _sudoku_units(n, box_h, box_w):
    """Yield the cells in each row, column, and box (1-indexed)."""
    for r in range(1, n + 1):
        yield [(r, c) for c in range(1, n + 1)]
    for c in range(1, n + 1):
        yield [(r, c) for r in range(1, n + 1)]
    for top in range(1, n + 1, box_h):
        for left in range(1, n + 1, box_w):
            yield [(r, c)
                   for r in range(top, top + box_h)
                   for c in range(left, left + box_w)]


def _peer_pairs(n, box_h, box_w):
    """Yield every unordered pair of peer cells exactly once."""
    seen = set()
    for unit in _sudoku_units(n, box_h, box_w):
        for i, first_cell in enumerate(unit):
            for second_cell in unit[i + 1:]:
                pair = tuple(sorted((first_cell, second_cell)))
                if pair not in seen:
                    seen.add(pair)
                    yield pair


def _prepare_definite_kb(kb):
    """Index a definite KB without changing its logical contents.

    The provided ``PropDefiniteKB.clauses_with_premise`` performs a full scan
    of the KB every time forward chaining processes a fact.  Sudoku contains
    thousands of Horn rules, so that implementation detail otherwise makes a
    single query take many seconds.  The index keeps the public KB type and
    the supplied ``pl_fc_entails`` algorithm unchanged while making premise
    lookup constant-time.
    """
    premise_index = {}
    rules_by_conclusion = {}
    facts = set()

    for clause in kb.clauses:
        premises, conclusion = parse_definite_clause(clause)
        if premises:
            rules_by_conclusion.setdefault(conclusion, []).append(tuple(premises))
            for premise in premises:
                premise_index.setdefault(premise, []).append(clause)
        else:
            facts.add(conclusion)

    def indexed_clauses_with_premise(premise):
        return premise_index.get(premise, ())

    kb.clauses_with_premise = indexed_clauses_with_premise
    kb._sudoku_indexed_clause_count = len(kb.clauses)
    kb._sudoku_rules_by_conclusion = rules_by_conclusion
    kb._sudoku_facts = facts
    return kb


def build_general_kb(n, box_h, box_w, givens):
    """Return a PropKB encoding this n x n Sudoku's constraints plus the given
    cells, as general clauses.

    Parameters
    ----------
    n, box_h, box_w : int
    givens : dict[(int, int), int]

    Returns
    -------
    PropKB
    """
    kb = PropKB()

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            # Every cell has at least one value.
            kb.tell(associate('|', [atom('Is', r, c, v)
                                    for v in range(1, n + 1)]))

            # A cell cannot have two different values.
            for v in range(1, n + 1):
                for w in range(v + 1, n + 1):
                    kb.tell(~atom('Is', r, c, v) | ~atom('Is', r, c, w))

    # Two cells in one row, column, or box cannot share a value.
    for ((r1, c1), (r2, c2)) in _peer_pairs(n, box_h, box_w):
        for v in range(1, n + 1):
            kb.tell(~atom('Is', r1, c1, v) |
                    ~atom('Is', r2, c2, v))

    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))
    return kb


def build_definite_kb(n, box_h, box_w, givens):
    """Return a PropDefiniteKB encoding this n x n Sudoku's constraints plus
    the given cells, using elimination + last-candidate reasoning.

    Parameters
    ----------
    n, box_h, box_w : int
    givens : dict[(int, int), int] -- {(row, col): value}, 1-indexed

    Returns
    -------
    PropDefiniteKB
    """
    kb = PropDefiniteKB()

    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                current = atom('Is', r, c, v)

                # If this cell is v, every other value is excluded.
                for w in range(1, n + 1):
                    if w != v:
                        kb.tell(Expr('==>', current,
                                     atom('Not', r, c, w)))

                # If every other value is excluded, this cell is v.
                exclusions = [atom('Not', r, c, w)
                              for w in range(1, n + 1) if w != v]
                if exclusions:
                    kb.tell(Expr('==>', associate('&', exclusions),
                                 current))
                else:
                    kb.tell(current)

    # Eliminate v from every peer of a cell known to contain v.
    for ((r1, c1), (r2, c2)) in _peer_pairs(n, box_h, box_w):
        for v in range(1, n + 1):
            kb.tell(Expr('==>', atom('Is', r1, c1, v),
                         atom('Not', r2, c2, v)))
            kb.tell(Expr('==>', atom('Is', r2, c2, v),
                         atom('Not', r1, c1, v)))

    return _prepare_definite_kb(kb)


class _IndexedForwardRun:
    """Index Horn premises and record the supplied FC algorithm's closure."""

    def __init__(self, kb):
        self.clauses = kb.clauses
        self.inferred = set()
        self._by_premise = {}
        for clause in self.clauses:
            if clause.op == '==>':
                for premise in conjuncts(clause.args[0]):
                    self._by_premise.setdefault(premise, []).append(clause)

    def clauses_with_premise(self, premise):
        self.inferred.add(premise)
        return self._by_premise.get(premise, ())


def solve_full_grid_fc(n, box_h, box_w, givens):
    """Solve the whole puzzle using build_definite_kb + pl_fc_entails.

    Returns
    -------
    dict[(int, int), int] -- {(row, col): value} for every cell
    """
    kb = build_definite_kb(n, box_h, box_w, givens)
    run = _IndexedForwardRun(kb)

    # The sentinel cannot be concluded by any Sudoku rule.  The provided
    # pl_fc_entails therefore runs to its fixed point once, while the indexed
    # adapter records every fact it processes.  This uses the supplied FC
    # implementation without repeating the same closure for every candidate.
    pl_fc_entails(run, expr('ForwardChainingClosureSentinel'))

    solved = {}

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            values = [
                v for v in range(1, n + 1)
                if atom('Is', r, c, v) in run.inferred
            ]
            if len(values) != 1:
                raise ValueError(
                    f'expected one entailed value for cell ({r}, {c}); '
                    f'found {values}'
                )
            solved[(r, c)] = values[0]

    return solved


class _BackwardState:
    """Cycle-safe table for goal-directed inference on one immutable KB."""

    def __init__(self, kb, signature):
        self.signature = signature
        self.facts = set()
        self.rules_by_head = {}
        for clause in kb.clauses:
            premises, conclusion = parse_definite_clause(clause)
            if premises:
                self.rules_by_head.setdefault(conclusion, []).append(
                    tuple(premises)
                )
            else:
                self.facts.add(conclusion)

        # known contains final truth values for fully analysed goals.  proof
        # stores a fact as None or the premises of its first grounded proof.
        self.known = {}
        self.proof = {fact: None for fact in self.facts}

    def entails(self, query):
        if query in self.known:
            return self.known[query]

        # Prove the query from facts by recursively proving every premise of
        # a candidate rule.  An expanded goal encountered again is unresolved
        # on that branch; a circular rule is not evidence for itself.
        expanded = set()
        deferred = []
        relevant_rules = []
        proven = self.facts.copy()

        def prove(goal, depth):
            if goal in proven:
                return True
            if goal in expanded:
                return False
            if depth >= 64:
                deferred.append(goal)
                return False

            expanded.add(goal)
            for premises in self.rules_by_head.get(goal, ()):
                relevant_rules.append((premises, goal))
                # Evaluate every premise, even if one has not been proved
                # yet, so cyclic dependencies remain available to the table.
                results = [prove(premise, depth + 1) for premise in premises]
                if all(results):
                    proven.add(goal)
                    self.proof.setdefault(goal, premises)
                    return True
            return False

        prove(query, 0)
        while deferred:
            goal = deferred.pop()
            if goal not in expanded:
                prove(goal, 0)

        # Settle only the rules reached by recursive goal expansion.  This
        # resolves mutually dependent goals after facts have been found and
        # permits safe caching of a negative answer.  It also keeps proofs
        # grounded in given facts rather than accepting unsupported cycles.
        waiting = {}
        remaining = []
        for rule_index, (premises, _head) in enumerate(relevant_rules):
            pending = set(premises) - proven
            remaining.append(pending)
            if not pending:
                continue
            for premise in pending:
                waiting.setdefault(premise, []).append(rule_index)

        agenda = list(proven.intersection(expanded))
        for rule_index, (premises, head) in enumerate(relevant_rules):
            if not remaining[rule_index] and head not in proven:
                proven.add(head)
                self.proof.setdefault(head, premises)
                agenda.append(head)

        for premise in agenda:
            for rule_index in waiting.get(premise, ()):
                pending = remaining[rule_index]
                pending.discard(premise)
                if pending:
                    continue
                rule_premises, head = relevant_rules[rule_index]
                if head in proven:
                    continue
                proven.add(head)
                self.proof.setdefault(head, rule_premises)
                agenda.append(head)

        self.known.update((goal, goal in proven) for goal in expanded)
        return query in proven


def _backward_state(kb):
    """Return cached tables, rebuilding them if the KB clauses changed."""
    signature = tuple(kb.clauses)
    state = getattr(kb, '_sudoku_backward_state', None)
    if state is None or state.signature != signature:
        state = _BackwardState(kb, signature)
        kb._sudoku_backward_state = state
    return state


def pl_bc_entails(kb, query):
    """Your own backward-chaining implementation.

    Parameters
    ----------
    kb : PropDefiniteKB
    query : Expr

    Returns
    -------
    bool
    """
    return bool(_backward_state(kb).entails(query))


def _atom_parts(sentence):
    """Parse a Sudoku atom without importing anything beyond course files."""
    name = sentence.op
    if name.startswith('Is'):
        kind, coordinates = 'Is', name[2:]
    elif name.startswith('Not'):
        kind, coordinates = 'Not', name[3:]
    else:
        return None

    parts = coordinates.split('_')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    r, c, v = map(int, parts)
    return kind, r, c, v


def _plain_atom(sentence):
    parts = _atom_parts(sentence)
    if parts is None:
        return str(sentence)
    kind, r, c, v = parts
    if kind == 'Is':
        return f'cell ({r}, {c}) contains {v}'
    return f'cell ({r}, {c}) cannot contain {v}'


def _plain_step(conclusion, premises):
    """Translate an actual proof edge into a human-readable Sudoku step."""
    target = _atom_parts(conclusion)
    sources = [_atom_parts(premise) for premise in premises]

    if (target is not None and target[0] == 'Not' and len(premises) == 1
            and sources[0] is not None and sources[0][0] == 'Is'):
        _, r, c, excluded = target
        _, source_r, source_c, source_value = sources[0]
        if (source_r, source_c) == (r, c):
            reason = (
                f'Cell ({r}, {c}) is already {source_value}, so it cannot '
                f'also contain {excluded}.'
            )
        elif source_r == r:
            reason = (
                f'Cell ({source_r}, {source_c}) contains {source_value}; '
                f'row {r} therefore excludes {excluded} from ({r}, {c}).'
            )
        elif source_c == c:
            reason = (
                f'Cell ({source_r}, {source_c}) contains {source_value}; '
                f'column {c} therefore excludes {excluded} from ({r}, {c}).'
            )
        else:
            reason = (
                f'Cell ({source_r}, {source_c}) contains {source_value}; '
                f'the shared box excludes {excluded} from ({r}, {c}).'
            )
        return f'Eliminate {excluded} from ({r}, {c})', reason

    if (target is not None and target[0] == 'Is' and sources
            and all(source is not None and source[0] == 'Not'
                    for source in sources)):
        _, r, c, value = target
        excluded = ', '.join(map(str, sorted(source[3] for source in sources)))
        return (
            f'Fix ({r}, {c}) as {value}',
            f'Values {excluded} have all been eliminated from cell ({r}, {c}); '
            f'{value} is its only remaining candidate.',
        )

    because = '; '.join(_plain_atom(premise) for premise in premises)
    return (
        f'Derive {_plain_atom(conclusion)}',
        f'Because {because}, infer that {_plain_atom(conclusion)}.',
    )


def _proof_steps(state, target):
    """Return a stored proof DAG in premise-before-conclusion order."""
    if target not in state.proof:
        return []

    emitted = set()
    ordered = []
    stack = [(target, False)]
    while stack:
        goal, ready = stack.pop()
        if goal in emitted:
            continue
        premises = state.proof.get(goal)
        if ready or premises is None:
            emitted.add(goal)
            if premises is None:
                ordered.append({
                    'title': f'Given: {_plain_atom(goal)}',
                    'explanation': f'The puzzle states that {_plain_atom(goal)}.',
                })
            else:
                title, explanation = _plain_step(goal, premises)
                ordered.append({'title': title, 'explanation': explanation})
            continue

        stack.append((goal, True))
        for premise in reversed(premises):
            if premise not in emitted:
                stack.append((premise, False))
    return ordered


def pl_bc_entails_with_trace(kb, query):
    """Return a Boolean verdict plus its genuine backward-chaining proof."""
    state = _backward_state(kb)
    entailed = bool(state.entails(query))
    trace_target = query

    # For a rejected value, explaining a provable explicit Not atom is more
    # useful than showing only that the requested Is atom lacks a proof.
    query_parts = _atom_parts(query)
    if not entailed and query_parts is not None and query_parts[0] == 'Is':
        _, r, c, v = query_parts
        excluded_query = atom('Not', r, c, v)
        if state.entails(excluded_query):
            trace_target = excluded_query

    steps = _proof_steps(state, trace_target)
    if entailed:
        explanation = (
            f'There is a fact-grounded rule chain proving that '
            f'{_plain_atom(query)}.'
        )
    elif trace_target != query:
        explanation = (
            f'The knowledge base instead proves that {_plain_atom(trace_target)}, '
            f'so the requested Is proposition is not entailed.'
        )
    else:
        explanation = (
            f'No fact-grounded rule chain proves that {_plain_atom(query)}.'
        )
    steps.append({
        'title': f'Verdict: {entailed}',
        'explanation': explanation,
    })
    return entailed, steps


def solve_full_grid_bc(n, box_h, box_w, givens):
    """Solve the whole puzzle using build_definite_kb + your own pl_bc_entails.

    For each cell, try each candidate value until pl_bc_entails confirms one
    -- the same per-cell strategy as solve_full_grid_fc, but backed by
    backward chaining instead of a single shared forward-chaining pass.

    Returns
    -------
    dict[(int, int), int] -- {(row, col): value} for every cell
    """
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = {}

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                if pl_bc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break
            else:
                raise ValueError(
                    f'backward chaining could not determine cell ({r}, {c})'
                )

    return solved
