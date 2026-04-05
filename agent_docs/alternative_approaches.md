# Alternative Approaches

Different ways to solve the same core problem: making structured NLP/ML text data readable during development.

## Pretty printer (non-interactive)

Think `bat` or `glow`, not `htop`. Non-interactive, just pipes beautiful structured output to stdout. `peep file.jsonl` dumps formatted records with color and structure, then exits. Composable with `less`, `head`, grep. Lower barrier to adoption since there's nothing to "learn" -- it's just a better `cat` for JSONL. Tradeoff: you lose navigation, search, and the ability to handle large files gracefully.

## Python library with a `show()` function

Instead of a separate tool, meet people where they already are -- the REPL, the script, the notebook. `from nlpeep import show; show(results)` renders rich output inline. Works in IPython, regular Python REPL, and even Jupyter (where it could render HTML). The insight: people don't always have a file yet. Often the data is in a variable. The rendering core and schema detection are reusable from the TUI. Tradeoff: you're now maintaining rendering for multiple output targets.

## Static HTML report generator

`peep file.jsonl -o report.html` produces a single self-contained HTML file. No server, no dependencies to view it -- just open it in a browser. Shareable with teammates who aren't in the terminal. This is how tools like `pytest-html` and `allure` work. Tradeoff: you've left the terminal, but the generation is still terminal-native.

## VS Code / editor extension

Render JSONL files with structure-aware formatting directly in the editor. Huge potential audience since most people have VS Code open alongside their terminal anyway. Tradeoff: completely different tech stack (TypeScript), and you're coupling to an editor ecosystem.

## NLP-aware `jq`

A query/filter DSL that understands NLP data semantics. `peep file.jsonl --where "metrics.f1 < 0.5" --show query,response,metrics` -- somewhere between jq and a viewer. Unix-composable. Tradeoff: you're building a query language, which is a much harder design problem.

## How these relate to the TUI

These aren't mutually exclusive. The pretty-printer and the Python library both share the same rendering core and schema detection -- the hard part already built in the TUI -- with different frontends. The HTML report is a natural export from any of them.

The `show()` library is a natural funnel: people who like it in their REPL may eventually want the full TUI for bigger files. The VS Code extension reaches the largest audience but is the most divergent in implementation.
