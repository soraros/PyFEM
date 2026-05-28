# v3 session workflow

Plan files live in the repo—not in chat. Branch: `v3`. Legacy `pyfem/` on `main` is out of scope unless the plan says otherwise.

## Plan files (pick one per session)

| File | Role |
|------|------|
| [roadmap.md](roadmap.md) | **Default.** Phases P0–P8 and checkboxes; agent picks the next open item |
| [feature-parity.md](feature-parity.md) | v1 vs v3 matrix; use when scoping “what’s missing” before coding |
| [parity.md](parity.md) | Skim layers A/B/C; read when adding or extending a parity case |

Supporting context (read as needed): [AGENTS.md](AGENTS.md), [architecture.md](architecture.md), [conventions.md](conventions.md).

## What you say (one line)

Replace `<plan>` with a path under `.agents/v3/` (usually `roadmap.md`).

```text
Work on the next item per <plan>.
```

Variants:

```text
Work on the next open checkbox in .agents/v3/roadmap.md.
```

```text
Work on phase P2 per .agents/v3/roadmap.md — only unchecked items in that phase.
```

```text
Work on “MPC / multi-point ties” per .agents/v3/roadmap.md.
```

```text
Continue v3: next skim + parity per .agents/v3/roadmap.md and .agents/v3/parity.md.
```

No need to paste the long template below unless you want to override scope.

## What the agent does (workflow)

When you use any one-liner above, the agent **must** run this sequence:

1. **Read the plan file** you named (default: `.agents/v3/roadmap.md`).
2. **Resolve the target item**
   - *Next item:* first `- [ ]` checkbox, top to bottom, in the earliest phase that still has open items.
   - *Named phase (e.g. P2):* first open checkbox only under that phase heading.
   - *Named goal:* match the checkbox text; if ambiguous, ask once.
3. **Read** [feature-parity.md](feature-parity.md) row for that item and [parity.md](parity.md) if the exit criteria need a new or updated skim.
4. **Implement** only that item; do not start the following checkbox or phase unless you asked for more.
5. **Verify**
   ```bash
   uv sync --group v3
   uv run pytest test/v3 -q
   uv run ruff check pyfem/v3 test/v3 --config pyfem/v3/ruff.toml
   ```
6. **Update the plan file(s):** mark `- [x]` on completed items; update [feature-parity.md](feature-parity.md) status column; add skim + `test/v3/test_parity_*.py` per [parity.md](parity.md) when applicable.
7. **Report:** item done, commands run, and the **next** open checkbox (phase + text) for your following session.

### Exit criteria (from roadmap)

An item is done only when its phase exit rule is met—typically:

- Green `pytest test/v3` including parity vs legacy `InputRead` + `LinearSolver` for any new/changed skim, or
- Explicit doc-only deliverable when the checkbox is documentation-only.

### Hard constraints (always)

- Python 3.13+; 2-space Ruff via `pyfem/v3/ruff.toml`
- Array-only `ProblemDefinition`; no `eval` in I/O
- Numba rules in root `AGENTS.md` (gitignored locally)
- No GUI/VTK, no in-place legacy replacement on `main`, no `.vscode/settings.json` or `notebooks/v3/*` in FEM commits unless you ask

## Expanded prompt (optional override)

Use when you need to narrow or widen a single session beyond “next item”:

```markdown
PyFEM v3 — branch `v3`.

**Plan:** `.agents/v3/roadmap.md` (or path you gave)
**Target:** [next open item | phase Pn | exact checkbox text]
**Also read:** `.agents/v3/feature-parity.md`, `.agents/v3/parity.md` if skims involved

Follow the agent workflow in `.agents/v3/WORKFLOW.md` (resolve target → implement → verify → update plan → report next item).

**Out of scope this session:** [optional list]
```

## Phase quick reference

| Phase | Open focus (see roadmap for truth) |
|-------|-------------------------------------|
| P1 | — (complete) |
| P2 | Plane strain, 3D continuum, `PatchTest8_3D` skim |
| P3+ | Nonlinear, structures, materials — see [feature-parity.md](feature-parity.md) |

## Parity discipline

- Do not loosen `parity.toml` without a note in [parity.md](parity.md).
- Prefer reusing book `.dat` via skim layer A (`input = "../../examples/..."`).
