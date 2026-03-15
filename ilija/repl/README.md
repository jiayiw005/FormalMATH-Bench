# Modified version of Lean REPL

Build using `lake exe cache get` and `lake build repl`.
Run using `lake exe repl <filepath>`. Imports mathlib by default (this will take a bit of time, around a minute on my machine). You can point it to one of our generated llm proof jsons that has been preprocessed with extract_body.py to extract the clean lean code with mathlib imports stripped. It will output a verify-llmname.json file that looks something like this:

```json
[{"answers":
  [{"verified": false,
    "sorries": 0,
    "messages":
    [{"severity": "error",
      "pos": {"line": 10, "column": 62},
      "endPos": {"line": 12, "column": 15},
      "data":
      "unsolved goals\nn : ℕ\nhn : 0 < n\nP : Fin n → ℝ\nhP : ∀ (i : Fin n), P i ∈ Set.Icc 0 1\n⊢ True ∧ True"}]},
   {"verified": true,
    "sorries": 0,
    "messages":
    [{"severity": "warning",
      "pos": {"line": 5, "column": 35},
      "endPos": {"line": 5, "column": 37},
      "data":
      "unused variable `hn`\n\nNote: This linter can be disabled with `set_option linter.unusedVariables false`"},
     {"severity": "warning",
      "pos": {"line": 6, "column": 5},
      "endPos": {"line": 6, "column": 7},
      "data":
      "unused variable `hP`\n\nNote: This linter can be disabled with `set_option linter.unusedVariables false`"}]}]},
 {"answers":...]
```
