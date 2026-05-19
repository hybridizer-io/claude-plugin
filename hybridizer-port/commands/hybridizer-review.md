---
name: hybridizer-review
description: Review pending kernel changes against the known Hybridizer gotchas list (MathF, AtomicExpr, OMP wrapper race, FloatResidentArray init, stale satellite traps). Returns findings; does not modify code.
---

Delegate to the **hybridizer-reviewer** subagent to vet kernel-related changes before they go through the build.

`$ARGUMENTS` may be:

- **Empty** — review pending changes (uncommitted diff). Default invocation, expected to be the most common.
- A **file path** — review that file specifically.
- A **commit range** (e.g. `HEAD~3..HEAD`) — review the changes in that range.

## What to do

1. **Determine the review target:**
   - Empty arguments → `git status` + `git diff` (unstaged) + `git diff --staged`. If both are empty, ask the user what they want reviewed.
   - File path → confirm it exists, read it.
   - Commit range → `git diff <range>`.

2. **Invoke the `hybridizer-reviewer` subagent** with the resolved target. Hand it the diff or file path; it does its own reading.

3. **Relay the subagent's findings report verbatim.** The format is:
   - ✗ hard findings (fix before building)
   - ⚠ soft findings (worth addressing)
   - ✓ confirmed good
   - Not checked

4. **Offer follow-ups if there are hard findings:**
   - "Want me to fix `<file:line>`?" — only if the fix is mechanical (e.g. `MathF.Exp` → `(float)System.Math.Exp`).
   - "Want me to run `/hybridizer-profile` after the fixes?"

5. **If there are no findings** ("No hard or soft findings; everything reads as correct"), say so plainly. Don't pad with congratulations.

## Don't

- Apply fixes from this command. The reviewer is read-only; the user opts into fixes explicitly.
- Run the build. If the user wants to confirm a clean build after fixes, they invoke that separately (or use the `hybridizer-builder` subagent directly).
- Filter the reviewer's findings. Pass them through verbatim, even if some look low-priority — the user decides what to address.
