---
name: hybridizer-log
description: Toggle the opt-in Hybridizer porting log. When on, every user prompt is appended to porting-to-hybridizer.md in the current project. Off by default.
---

Toggle the project-local Hybridizer porting log.

`$ARGUMENTS` should be one of `on`, `off`, or `status`. If empty, default to `status`.

## What to do

The log is gated by a flag file `.hybridizer-log-enabled` at the **project root** (i.e. the current working directory when Claude was launched). The plugin's `UserPromptSubmit` hook checks this file on every prompt and skips logging if absent.

### `on`

1. Create `.hybridizer-log-enabled` in the current working directory (empty file is fine; presence is what matters).
2. Add `.hybridizer-log-enabled` and `porting-to-hybridizer.md` to `.gitignore` if not already present, **only if the working directory is a git repository**. Don't pollute non-repo directories with a `.gitignore` write.
3. Report: "Porting log enabled. Every prompt from now on will be appended to `porting-to-hybridizer.md` in this directory."

### `off`

1. Delete `.hybridizer-log-enabled` if it exists.
2. Report: "Porting log disabled. `porting-to-hybridizer.md` is preserved; the hook just stops appending to it."

### `status`

1. Check whether `.hybridizer-log-enabled` exists in the current working directory.
2. If yes: report "Porting log: ON. Appending to `<cwd>/porting-to-hybridizer.md`."
3. If no: report "Porting log: OFF. (Run `/hybridizer-log on` to enable.)"
4. If `porting-to-hybridizer.md` exists, mention its size as a heads-up.

## Don't

- Append to the log yourself in this command — the hook handles writes. This command only toggles the gate.
- Move the flag file location. Other parts of the plugin (the hook) assume it's at `$CLAUDE_PROJECT_DIR/.hybridizer-log-enabled`.
- Modify `porting-to-hybridizer.md` itself except via the hook. If the user wants to clear it, they can `rm` or truncate manually.
