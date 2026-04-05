# TUI Inspection

Run the nlpeep TUI headlessly on data files and inspect the widget tree for rendering issues. Use this after making changes to detection, rendering, or layout. Checks are structural and generalize to any dataset.

## How to run

Write and execute a Python script using Textual's `run_test()` API. The script should accept a directory or list of files, load each one, and report issues found via widget tree inspection.

Run on all files in `tests/reference_data/` and `tests/fixtures/` at minimum, but the checks apply to any file the user provides.

## Checks to perform

All checks are generic -- they examine widget properties, not dataset-specific content.

### 1. Detection coverage
- At least one non-UNMAPPED role should be detected. If zero roles are detected, the file is effectively unrecognized.
- At least one of QUERY or INPUT should be detected for most NLP data. Flag as a warning (not error) since some datasets are purely metric tables.

### 2. Content truncation
- Any Static widget whose content length exceeds 200 characters but whose rendered height is 1 line is likely truncated. The content is there but the user can't see it.
- The `#query-header` widget has `max-height: 6`. If the text placed there exceeds what 6 lines can display at the terminal width, it's overflowing. Estimate: if `len(text) > terminal_width * 5`, flag it.

### 3. Empty panes
- Tab panes with no child widgets (or only whitespace content) indicate a role was detected but the value didn't resolve or render. This usually means a path resolution failure.

### 4. Role-value mismatches
- A field mapped to QUERY or INPUT whose value is not a string (e.g., int, list, dict) will render poorly.
- A field mapped to DOCUMENTS whose value is not a list will not render as doc cards.
- A field mapped to METRICS whose value is not a number or dict of numbers will not render as score bars.

### 5. Scroll containment
- Long text (LONG_TEXT, MARKDOWN) should be inside a scrollable ancestor. Walk up the parent chain and check for VerticalScroll or a widget with `overflow_y: auto/scroll`.

### 6. Record navigation
- Navigate through multiple records (press `j` a few times). Verify the content area actually updates (the query header text or tab content should change between records).

### 7. Trace assembly (when applicable)
- If the original file has more records than the store after loading (trace assembly fired), verify the composite record has `_trace_spans` and it contains all original spans.

## Template script

```python
import asyncio
from pathlib import Path
from nlpeep.app import NLPeepApp
from nlpeep.widgets.field_panel import FieldPanel
from textual.widgets import Static, TabPane

async def inspect_file(path: Path, terminal_width: int = 120) -> list[str]:
    """Inspect a file and return a list of issue descriptions."""
    issues = []
    try:
        app = NLPeepApp(path=path)
    except Exception as e:
        return [f"Failed to create app: {e}"]

    async with app.run_test(size=(terminal_width, 40)) as pilot:
        await pilot.pause()

        if not app._store or not app._store.records:
            return ["No records loaded"]

        mapping = app._mapping

        # 1. Detection coverage
        roles = {m.role.value for m in mapping.mappings if m.role.value != "unmapped"}
        if not roles:
            issues.append("DETECTION: no roles detected")
        elif "query" not in roles and "input" not in roles:
            issues.append("DETECTION: no query/input role (may be expected for metric-only data)")

        # 2. Content truncation
        for w in app.query(Static):
            text = str(getattr(w, "renderable", ""))
            if len(text) > 200 and w.size.height <= 1:
                classes = list(getattr(w, "classes", []))
                issues.append(
                    f"TRUNCATION: {len(text)} chars in 1-line Static "
                    f"(classes={classes}, first 60 chars: {text[:60]!r})"
                )

        # Query header overflow
        try:
            header = app.query_one("#query-header", Static)
            header_text = str(header.renderable)
            max_chars = terminal_width * 5
            if len(header_text) > max_chars:
                issues.append(
                    f"HEADER OVERFLOW: {len(header_text)} chars in header "
                    f"(max ~{max_chars} visible)"
                )
        except Exception:
            pass

        # 3. Empty tab panes
        for pane in app.query(TabPane):
            children = [c for c in pane.query("*") if c is not pane]
            if not children:
                label = getattr(pane, "_label", pane.id or "unknown")
                issues.append(f"EMPTY TAB: {label}")

        # 4. Role-value type mismatches
        record = app._store[0]
        for m in mapping.mappings:
            if m.role.value == "unmapped":
                continue
            val = record.get_path(m.json_path)
            if val is None:
                continue
            role = m.role.value
            if role in ("query", "input") and not isinstance(val, str):
                issues.append(f"TYPE MISMATCH: {m.json_path} ({role}) is {type(val).__name__}, expected str")
            if role == "documents" and not isinstance(val, list):
                issues.append(f"TYPE MISMATCH: {m.json_path} (documents) is {type(val).__name__}, expected list")

        # 5. Navigation works
        try:
            header_before = str(app.query_one("#query-header", Static).renderable)
            await pilot.press("j")
            await pilot.pause()
            header_after = str(app.query_one("#query-header", Static).renderable)
            if len(app._store) > 1 and header_before == header_after:
                issues.append("NAVIGATION: pressing j did not change displayed record")
        except Exception:
            pass

    return issues


async def main():
    import sys
    if len(sys.argv) > 1:
        dirs_or_files = [Path(p) for p in sys.argv[1:]]
    else:
        dirs_or_files = [Path("tests/reference_data"), Path("tests/fixtures")]

    # Collect files: expand directories, accept individual files
    supported = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}
    paths = []
    for p in dirs_or_files:
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir() if f.is_file() and f.suffix in supported))
        elif p.is_file() and p.suffix in supported:
            paths.append(p)

    for path in paths:
        try:
            issues = await inspect_file(path)
            status = "PASS" if not issues else "FAIL"
            print(f"{status:4s}  {path.name}")
            for issue in issues:
                print(f"        {issue}")
        except Exception as e:
            print(f"ERR   {path.name}: {e}")

asyncio.run(main())
```

## When to use

- After changing renderers, field panels, record view, or CSS
- After adding new archetypes or roles
- After modifying schema detection logic
- When adding new reference datasets to verify they render well
- Before releases to catch regressions
