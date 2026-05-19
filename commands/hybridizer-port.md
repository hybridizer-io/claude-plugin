---
name: hybridizer-port
description: Analyse a C# function and propose a Hybridizer kernel shape (parallelism, data lifetime, atomics, per-flavor split, gotchas). Returns a structured proposal — does not write the kernel.
---

Delegate to the **hybridizer-porter** subagent to produce a kernel-shape proposal for the target function.

The user invokes this as `/hybridizer-port <file-or-method-or-selection>`. `$ARGUMENTS` may be:

- A **file path** (e.g. `Math/RmsNorm.cs`) — port the most relevant function in the file, asking which if there are several candidates.
- A **file:line range** (e.g. `Math/RmsNorm.cs:42-78`) — port the specified slice.
- A **fully-qualified method name** (e.g. `LlamaCsharp.Math.RmsNorm.ScaleInPlace`) — port that method specifically.
- **Empty** — ask the user which function they want to port, OR pick the function currently visible in their workspace if obvious from context.

## What to do

1. **Resolve the target.** If ambiguous, ask once for clarification before launching the subagent — sending a porter with a vague target wastes its pass.
2. **Confirm the target lives in a project that already has Hybridizer wired in.** Check for `<CompileCUDA>` in the `.csproj` and `$(HybridizerTool)` in `Directory.Build.props`. If absent, suggest running `/hybridizer-init` first.
3. **Invoke the `hybridizer-porter` subagent** with the resolved target. Pass enough context for it to understand the call sites — at minimum, the target file path and method signature.
4. **Relay the subagent's structured proposal** to the user verbatim.
5. **Offer the obvious follow-ups:**
   - "Want me to write the `[EntryPoint]` based on this proposal?"
   - "Want me to run `/hybridizer-review` once the kernel is drafted?"
   - "Want me to set up profiling for the resulting build with `/hybridizer-profile`?"

## Don't

- Write the kernel yourself in this command — the porter returns a proposal, not code. Writing is a separate step the user opts into.
- Skip the prerequisite check. Porting into a project without `<CompileCUDA>` produces a proposal that can't be built.
- Invent perf numbers. The porter qualifies expected wins as expectations, not measured tok/s.
