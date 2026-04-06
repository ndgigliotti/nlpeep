# nlpeep

A terminal-native viewer that makes structured NLP and ML data readable.

## Install

```bash
uv pip install nlpeep
```

or

```bash
pip install nlpeep
```

Requires Python 3.11+.

## Usage

```bash
peep results.jsonl
```

Point `peep` at any supported data file and it renders records with
structure-aware widgets. Fields are auto-detected into roles (query, response,
documents, metrics, etc.) so it works without configuration.

Supported formats: **JSONL**, **JSON**, **CSV**, **TSV**, **Parquet**.

## What it detects

nlpeep auto-maps fields into roles, and each role gets a purpose-built renderer:

| Role | What gets matched | Rendering |
|------|-------------------|-----------|
| Query | prompts, questions, source text | Formatted text, markdown |
| Input | raw input fields | Formatted text |
| Response | completions, summaries, translations | Formatted text, markdown |
| Ground Truth | labels, reference answers, gold standard | Side-by-side with response |
| Documents | retrieved passages, context chunks | Cards with score, source, rank |
| Metrics | scores, eval results, confidence | Color-coded bars, numeric display |
| Trace | LLM call chains, tool use, steps | Collapsible tree with timing |
| ID | record identifiers | Compact display |
| Metadata | timestamps, config, tags | Key-value display |

Any field that does not match a known role is shown under a generic "Details"
section.

## Key bindings

| Key | Action |
|-----|--------|
| `j` / `k` | Next / previous record |
| `Ctrl+f` | Focus search |
| `Escape` | Unfocus search |
| `m` | Open field mapping editor |
| `Ctrl+r` | Reload file from disk |
| `q` | Quit |

## Configuration

Auto-detection handles the common case. When it gets a field wrong, press `m`
to open the mapping editor and reassign roles. The corrected mapping is saved
as a `.nlpeep.toml` file next to your data so it persists across sessions.

Config files are discovered in this order:

1. Explicit `--config path/to/config.toml`
2. Per-file: `.nlpeep.<filename>.toml` in the same directory
3. Directory: `.nlpeep.toml` in the same directory
4. CWD: `.nlpeep.toml` in the current working directory

Example `.nlpeep.toml`:

```toml
[mapping]
query = "question"
response = "answer"
documents = "contexts"
metrics = ["f1_score", "precision", "recall"]
```

## Supported formats

JSONL, JSON (array-of-objects), CSV, TSV, and Parquet. Data is loaded via
Polars, so large files are handled efficiently.

For multi-span pipeline trace data (LangSmith, Langfuse, Phoenix), nlpeep
assembles spans into a single trace tree per record so you can inspect the full
call chain in one view.
