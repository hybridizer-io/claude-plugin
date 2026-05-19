# hybridizer claude-plugin

A Claude Code plugin that helps migrate existing C# codebases to CUDA using [Altimesh Hybridizer](https://github.com/hybridizer-io/hybridizer-basic-samples).

The plugin bundles:

- **A `hybridizer-port` skill** — Hybridizer attributes, device-code restrictions, the build pipeline, host launch idioms, kernel patterns (reductions, cooperative blocks, warp shuffle, CUB), CUDA graph capture, perf tuning, and a curated gotchas list.
- **Three subagents** — `hybridizer-porter` (analyse a C# function and propose a kernel shape), `hybridizer-reviewer` (check a diff against the gotchas list), `hybridizer-builder` (invoke `Hybridizer.Application` and parse errors).
- **Slash commands** — `/hybridizer-init`, `/hybridizer-port`, `/hybridizer-review`, `/hybridizer-profile`, `/hybridizer-log`.
- **An opt-in hook** that appends every porting prompt to `porting-to-hybridizer.md` in your project (off by default; toggle with `/hybridizer-log on`).

The skill is wired with progressive disclosure: `SKILL.md` is short and indexes deeper reference files in `hybridizer-port/skills/hybridizer-port/references/` that Claude pulls in only when a task touches that topic.

## Requirements

- **Claude Code** (any recent version with plugin / marketplace support).
- **Python 3** in `PATH` — used only by the optional porting-log hook (`/hybridizer-log on`). The rest of the plugin works without it.
- **Altimesh Hybridizer** install if you're actually porting code (the plugin's content references it; the plugin itself is install-light).

## Install

### From this marketplace (recommended)

```
/plugin marketplace add hybridizer-io/claude-plugin
/plugin install hybridizer-port@hybridizer
```

### From a local clone (for development)

```
git clone https://github.com/hybridizer-io/claude-plugin /path/to/claude-plugin
/plugin marketplace add /path/to/claude-plugin
/plugin install hybridizer-port@hybridizer
```

## Usage

The skill auto-loads when Claude detects you're working on Hybridizer code (CUDA porting context, references to `[Kernel]`/`[EntryPoint]`, `Hybridizer.Application` build issues, etc.). You can also invoke it explicitly with the slash commands above.

To enable the optional porting log:

```
/hybridizer-log on
```

This creates a `.hybridizer-log-enabled` flag in the current project; every prompt thereafter is appended to `porting-to-hybridizer.md`. Disable with `/hybridizer-log off`.

## Layout

```
claude-plugin/
├── .claude-plugin/
│   └── marketplace.json          (lists hybridizer-port, source: ./hybridizer-port)
├── hybridizer-port/              (the plugin itself)
│   ├── .claude-plugin/
│   │   └── plugin.json
│   ├── skills/hybridizer-port/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── methodology.md
│   │       ├── attributes.md
│   │       ├── device-code.md
│   │       ├── build-pipeline.md
│   │       ├── host-launch.md
│   │       ├── kernel-patterns.md
│   │       ├── reductions.md
│   │       ├── graph-capture.md
│   │       ├── perf-tuning.md
│   │       ├── gotchas.md
│   │       └── samples-index.md
│   ├── agents/
│   ├── commands/
│   └── hooks/
├── README.md
└── LICENSE
```

## Status

Early — content is distilled from a working port (TinyLlama Q8_0 inference, ~125 tok/s on RTX 5070). Expect iteration. Issues and PRs welcome.

## License

MIT (see [LICENSE](LICENSE)).
