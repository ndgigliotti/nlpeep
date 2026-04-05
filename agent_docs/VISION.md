# NLPeep Vision

## One-liner

A terminal-native viewer that makes structured text data readable.

## The problem

NLP and ML work produces enormous amounts of structured text data -- evaluation outputs, pipeline traces, labeled datasets, model comparisons, retrieval results. Reviewing this data is a core part of the workflow, and the existing options are all bad:

- **Jupyter notebooks** require port forwarding, a browser, and context switching. They're fine for writing code, terrible for reading data. Cells get cluttered. Scrolling through hundreds of records is painful.
- **Web apps** (Streamlit, Gradio, custom dashboards) don't mesh with remote development environments. Nobody launches a web server to glance at eval results. The ones that persist (Langfuse, LangSmith) require infrastructure and lock you into their ecosystem.
- **jq / cat / less** work everywhere but treat text as undifferentiated strings. You're squinting at raw JSON, mentally parsing structure, losing context as it scrolls past.
- **Copying files locally** to view them in a GUI is friction that interrupts flow.

The realistic workflow for most people is: SSH into a box, run a job, look at the output. There's no good "look at the output" tool for text-heavy structured data.

## The solution

NLPeep is a TUI that understands text data. You point it at a file and it renders records with structure-aware widgets -- markdown gets formatted, scores get color-coded bars, documents get cards with metadata, chat messages get role-colored blocks. It auto-detects what fields represent (queries, responses, labels, metrics, traces) so it works without configuration.

The key properties:

- **Terminal-native.** Works over SSH, in tmux, on any box with Python. No browser, no port forwarding, no infrastructure.
- **Zero-config.** Auto-detection means you run `peep file.jsonl` and it does something useful immediately. Configuration exists for when auto-detection is wrong, not as a prerequisite.
- **Read-optimized.** This is a viewer, not an editor or query engine. The UX is optimized for reading through records, comparing fields, and understanding what your pipeline produced. Keyboard-driven navigation, search, and sensible defaults for how much text to show.
- **Domain-aware.** Unlike generic JSON viewers, NLPeep knows what NLP data looks like. It uses that knowledge to choose renderers, group related fields, and surface what matters.

## Who this is for

Anyone who produces or consumes structured text data in a terminal environment:

- ML engineers debugging and evaluating pipelines (RAG, classification, summarization, translation, extraction, alignment)
- Researchers reviewing experiment outputs and dataset samples
- Data scientists inspecting labeled data and annotation quality
- Platform engineers building NLP infrastructure who need to verify data flowing through systems

The common thread is: you have structured records with text in them, and you want to read them comfortably without leaving your terminal.

## What NLPeep is not

- **Not a query engine.** Use jq, DuckDB, or pandas for filtering and aggregation. NLPeep is for reading the results.
- **Not a monitoring dashboard.** Use Langfuse, LangSmith, or Grafana for production observability. NLPeep is for development-time inspection.
- **Not an editor.** It doesn't modify your data.
- **Not a notebook replacement.** Notebooks are for writing code that produces data. NLPeep is for reading the data that code produces.

## Design principles

1. **Work immediately.** The first run on any file should produce a useful view. Require zero configuration for the common case.
2. **Respect the terminal.** No mouse-required interactions. Vim-style navigation. Works in 80-column terminals. Looks good in dark themes (which is all of them on remote boxes).
3. **Render text as text.** The whole point is making text readable. Markdown should look like markdown. Long passages should be scrollable, not truncated to one line. Scores should be visual, not just numbers.
4. **Stay small.** Minimal dependencies, fast startup, single entry point. This is a tool you reach for reflexively, not one you plan around.
5. **Support diverse formats.** NLP tooling is fragmented. NLPeep should handle output from any framework (LangChain, RAGAS, Phoenix, custom pipelines) without per-framework plugins.

## Supported data types

The auto-detection system maps fields into roles, and each role gets appropriate rendering:

| Role | Examples | Rendering |
|------|----------|-----------|
| Query / Input | prompts, questions, source text | Formatted text, markdown |
| Response / Output | completions, summaries, translations | Formatted text, markdown |
| Ground Truth | labels, reference answers, gold standard | Side-by-side with response |
| Documents | retrieved passages, context chunks | Cards with score, source, rank |
| Metrics | scores, eval results, confidence | Color-coded bars, numeric display |
| Trace | LLM call chains, tool use, steps | Collapsible tree with timing |
| Chat / Messages | conversation history, multi-turn | Role-colored message blocks |
| Metadata | IDs, timestamps, config, tags | Compact key-value display |

This list should grow as we encounter more NLP data patterns. The architecture supports adding new roles and renderers without restructuring.

## Current state (v0.0.1)

Working and usable for the core use case. Auto-detection handles RAG pipeline data well. The neon theme looks distinctive. Navigation and search work. Config persistence via `.nlpeep.toml` works.

Gaps: no README, no tests, no stdin support, schema detection skews toward RAG patterns and needs broadening for general NLP tasks. Published on PyPI to claim the name but not yet promoted.

## Near-term direction

The priority is making this installable and discoverable:

1. Broaden schema detection beyond RAG (classification, NER, summarization, translation patterns)
2. Tests for schema detection and rendering (the parts most likely to regress)
3. README with terminal recording showing the tool in action
4. stdin support (`some_command | peep`) for pipeline integration
5. Clipboard copy for field values
6. In-app help screen
