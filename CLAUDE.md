# nlpeep

A Textual-based TUI for inspecting NLP and RAG pipeline data.

## Quick reference

```bash
uv pip install -e ".[dev]"     # install in dev mode
uv run peep <file>             # run the app
uv run pytest                  # run tests
```

## Project layout

```
src/nlpeep/          # application source
  app.py             # Textual App subclass
  __main__.py        # CLI entry point (peep command)
  schema.py          # auto-detection of data schemas/archetypes
  data.py            # loaders for JSONL, JSON, CSV, TSV, Parquet
  config.py          # .nlpeep.toml config handling
  renderers.py       # rich rendering helpers
  widgets/           # Textual widgets (navigator, doc_card, etc.)
  styles/            # TCSS stylesheets
tests/               # pytest tests
  fixtures/          # small synthetic test data
  reference_data/    # real-world format samples
```

## Conventions

- Python >= 3.11, managed with `uv` (never pip).
- Pre-commit hooks enforce ruff lint/format, LF line endings, no large files.
- Commit messages: short imperative summary ("Add ...", "Fix ...").
