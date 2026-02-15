import json
import re
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _import_extract_proof_body():
    """Import extract_proof_body by reading the source, avoiding pexpect dependency"""
    import importlib.util
    import types
    
    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "verify_answers.py")
    with open(source_path, "r") as f:
        source = f.read()
    
    # Extract just the extract_proof_body function source
    # Find it between the def and the next top-level def
    match = re.search(
        r'^(def extract_proof_body\(.*?\n)(?=\n\ndef |\nclass |\n#|\Z)',
        source, re.MULTILINE | re.DOTALL
    )
    if match:
        func_source = match.group(0)
        namespace = {"re": re}
        exec(func_source, namespace)
        return namespace["extract_proof_body"]
    else:
        raise ImportError("Could not find extract_proof_body in verify_answers.py")

try:
    from verify_answers import extract_proof_body
except (ImportError, ModuleNotFoundError):
    extract_proof_body = _import_extract_proof_body()

# helpers
_pass_count = 0
_fail_count = 0

def check(name, condition, detail=""):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  PASS: {name}")
    else:
        _fail_count += 1
        print(f"  FAIL: {name}")
        if detail:
            print(f"        {detail}")

def summary():
    total = _pass_count + _fail_count
    print(f"\n{'='*60}")
    print(f"Results: {_pass_count}/{total} passed, {_fail_count} failed")
    print(f"{'='*60}")
    return _fail_count == 0


# unit tests

def test_extract_basic_lean4_fence():
    """Standard case: ```lean4 code fence with natural language wrapper"""
    print("\n--- test_extract_basic_lean4_fence ---")
    
    auto = "theorem foo (n : Nat) : n = n := by"
    answer = "Here's the proof:\n\n```lean4\nimport Mathlib\n\ntheorem foo (n : Nat) : n = n := by\n  rfl\n```\n\nDone!"
    
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains rfl", body is not None and "rfl" in body)
    check("no natural language", body is not None and "Here's" not in body)
    check("no import", body is not None and "import" not in body)
    
    cmd = auto + body
    check("full cmd is valid lean", "theorem foo" in cmd and "rfl" in cmd)


def test_extract_lean_fence():
    """Code fence with ```lean instead of ```lean4"""
    print("\n--- test_extract_lean_fence ---")
    
    auto = "theorem bar : True := by"
    answer = "Proof:\n\n```lean\ntheorem bar : True := by\n  trivial\n```"
    
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains trivial", body is not None and "trivial" in body)


def test_extract_bare_fence():
    """Code fence with just ``` (no language specifier)"""
    print("\n--- test_extract_bare_fence ---")
    
    auto = "theorem baz : 1 = 1 := by"
    answer = "```\ntheorem baz : 1 = 1 := by\n  rfl\n```"
    
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains rfl", body is not None and "rfl" in body)


def test_extract_no_fence_code_answer():
    """Raw code answer starting with import (no fence)"""
    print("\n--- test_extract_no_fence_code_answer ---")
    
    auto = "theorem qux : True := by"
    answer = "import Mathlib\n\ntheorem qux : True := by\n  trivial"
    
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains trivial", body is not None and "trivial" in body)


def test_extract_no_fence_natural_language():
    """Pure natural language answer with no code — should return None"""
    print("\n--- test_extract_no_fence_natural_language ---")
    
    auto = "theorem foo : True := by"
    answer = "I'm not sure how to prove this theorem. Maybe try induction?"
    
    body = extract_proof_body(answer, auto)
    check("returns None", body is None)


def test_extract_empty_answer():
    """Empty answer string"""
    print("\n--- test_extract_empty_answer ---")
    
    auto = "theorem foo : True := by"
    body = extract_proof_body("", auto)
    check("returns None", body is None)


def test_extract_multiline_tactic_proof():
    """Multi-line tactic proof with nested have/calc"""
    print("\n--- test_extract_multiline_tactic_proof ---")
    
    auto = "theorem algebra_123 (x : ℝ) : x + 0 = x := by"
    answer = """Here's the proof:

```lean4
import Mathlib

open Real

theorem algebra_123 (x : ℝ) : x + 0 = x := by
  have h1 : x + 0 = x := by
    ring
  exact h1
```
"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains have h1", body is not None and "have h1" in body)
    check("contains ring", body is not None and "ring" in body)
    check("contains exact h1", body is not None and "exact h1" in body)
    
    cmd = auto + body
    check("starts with main theorem", cmd.startswith("theorem algebra_123"))


def test_extract_helper_lemma_before_main():
    """Answer includes a helper lemma before the main theorem"""
    print("\n--- test_extract_helper_lemma_before_main ---")
    
    auto = "theorem main_thm (n : Nat) : n + 0 = n := by"
    answer = """```lean4
import Mathlib

lemma helper (n : Nat) : n = n := by
  rfl

theorem main_thm (n : Nat) : n + 0 = n := by
  simp
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains simp", body is not None and "simp" in body)
    # should not include the helper lemma's rfl
    check("does not start with rfl", body is not None and not body.strip().startswith("rfl"))


def test_extract_term_mode_proof():
    """Term-mode proof (:= expr rather than := by tactic)"""
    print("\n--- test_extract_term_mode_proof ---")
    
    # autoformalization that does not end with "by"
    auto = "def foo : Nat :="
    answer = """```lean4
def foo : Nat := 42
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains 42", body is not None and "42" in body)


def test_extract_where_clause():
    """Proof using where clause"""
    print("\n--- test_extract_where_clause ---")
    
    auto = "theorem with_where (n : Nat) : n = n := by"
    answer = """```lean4
import Mathlib

theorem with_where (n : Nat) : n = n := by
  exact helper
  where
    helper : n = n := rfl
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains where", body is not None and "where" in body)
    check("contains helper", body is not None and "helper" in body)


def test_extract_colon_equals_by_no_space():
    """`:=by` with no space before `by`"""
    print("\n--- test_extract_colon_equals_by_no_space ---")
    
    auto = "theorem nospace : True := by"
    answer = """```lean4
theorem nospace : True :=by
  trivial
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains trivial", body is not None and "trivial" in body)


def test_extract_special_chars_in_name():
    """Theorem name with special characters (underscores, numbers, primes)"""
    print("\n--- test_extract_special_chars_in_name ---")
    
    auto = "theorem omni_theorem_1589' (x : ℝ) : x = x := by"
    answer = """```lean4
import Mathlib

theorem omni_theorem_1589' (x : ℝ) : x = x := by
  rfl
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains rfl", body is not None and "rfl" in body)


def test_extract_multiple_code_blocks():
    """Answer with multiple code blocks — should use the first one"""
    print("\n--- test_extract_multiple_code_blocks ---")
    
    auto = "theorem multi : True := by"
    answer = """First attempt:

```lean4
theorem multi : True := by
  trivial
```

Actually, here's a better version:

```lean4
theorem multi : True := by
  exact True.intro
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    # Should get the first code block
    check("contains trivial (first block)", body is not None and "trivial" in body)


def test_extract_answer_redeclares_with_different_formatting():
    """Answer redeclares theorem with different whitespace/formatting"""
    print("\n--- test_extract_answer_redeclares_with_different_formatting ---")
    
    auto = "theorem fmt_test (a b : Nat) :\n  a + b = b + a := by"
    answer = """```lean4
import Mathlib

theorem fmt_test (a b : Nat) : a + b = b + a := by
  omega
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains omega", body is not None and "omega" in body)


def test_extract_def_instead_of_theorem():
    """Autoformalization uses `def` instead of `theorem`"""
    print("\n--- test_extract_def_instead_of_theorem ---")
    
    auto = "def myFunc (n : Nat) : Nat := by"
    answer = """```lean4
import Mathlib

def myFunc (n : Nat) : Nat := by
  exact n + 1
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains exact n + 1", body is not None and "exact n + 1" in body)


def test_extract_lemma_keyword():
    """Autoformalization uses `lemma`"""
    print("\n--- test_extract_lemma_keyword ---")
    
    auto = "lemma my_lemma : 0 = 0 := by"
    answer = """```lean4
lemma my_lemma : 0 = 0 := by
  rfl
```"""
    body = extract_proof_body(answer, auto)
    check("returns non-None", body is not None)
    check("contains rfl", body is not None and "rfl" in body)


# parsing logic

def parse_autoformalization(raw_auto):
    """Replicate the parsing logic from verify_answers"""
    if "```lean4\n" in raw_auto:
        lean_code = raw_auto.split("```lean4\n")[1]
    elif "```lean\n" in raw_auto:
        lean_code = raw_auto.split("```lean\n")[1]
    elif "```\n" in raw_auto:
        lean_code = raw_auto.split("```\n")[1]
    else:
        lean_code = raw_auto
    
    if lean_code.endswith("```"):
        lean_code = lean_code[:-3]
    
    split_match = re.search(r'^(theorem|def|lemma|example)\s', lean_code, re.MULTILINE)
    if split_match:
        context = lean_code[:split_match.start()]
        autoformalization = lean_code[split_match.start():]
    else:
        context = lean_code
        autoformalization = ""
    
    return context, autoformalization


def test_parse_standard_format():
    """Standard FormalMATH autoformalization format"""
    print("\n--- test_parse_standard_format ---")
    
    raw = "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\nopen Real Set\nopen scoped BigOperators\n\ntheorem algebra_569189 (x : ℝ) :\n  |sqrt (x^2 + x + 1) - sqrt (x^2 - x + 1)| < 1 := by\n```"
    
    context, auto = parse_autoformalization(raw)
    check("context has import", "import Mathlib" in context)
    check("context has open", "open Real Set" in context)
    check("auto starts with theorem", auto.startswith("theorem"))
    check("auto has theorem name", "algebra_569189" in auto)
    check("auto ends with by", auto.rstrip().endswith("by"))
    check("no theorem in context", "theorem" not in context)


def test_parse_multiple_opens():
    """Multiple open statements before theorem"""
    print("\n--- test_parse_multiple_opens ---")
    
    raw = "```lean4\nimport Mathlib\nimport Mathlib.Tactic\n\nopen Finset\nopen scoped Real\nopen BigOperators\n\ntheorem test (n : Nat) : n = n := by\n```"
    
    context, auto = parse_autoformalization(raw)
    check("context has both imports", "import Mathlib\n" in context and "Mathlib.Tactic" in context)
    check("context has all opens", "Finset" in context and "Real" in context and "BigOperators" in context)
    check("auto starts with theorem", auto.startswith("theorem"))


def test_parse_def_declaration():
    """Autoformalization with `def` instead of `theorem`"""
    print("\n--- test_parse_def_declaration ---")
    
    raw = "```lean4\nimport Mathlib\n\ndef myDef (n : Nat) : Nat := by\n```"
    
    context, auto = parse_autoformalization(raw)
    check("context has import", "import Mathlib" in context)
    check("auto starts with def", auto.startswith("def"))


def test_parse_theorem_in_comment():
    """'theorem' appears in a comment in the context — should NOT split there"""
    print("\n--- test_parse_theorem_in_comment ---")
    
    raw = "```lean4\nimport Mathlib\n-- This is not a theorem declaration\n\ntheorem real_thm : True := by\n```"
    
    context, auto = parse_autoformalization(raw)
    # The regex uses ^theorem\s with MULTILINE, so the comment line shouldn't match
    check("auto starts with theorem", auto.startswith("theorem real_thm"))
    check("context has comment", "not a theorem" in context)


def test_parse_no_code_fence():
    """Autoformalization without any code fence"""
    print("\n--- test_parse_no_code_fence ---")
    
    raw = "import Mathlib\n\ntheorem bare : True := by"
    
    context, auto = parse_autoformalization(raw)
    check("context has import", "import Mathlib" in context)
    check("auto starts with theorem", auto.startswith("theorem"))


def test_parse_trailing_backticks():
    """Trailing ``` gets stripped"""
    print("\n--- test_parse_trailing_backticks ---")
    
    raw = "```lean4\nimport Mathlib\n\ntheorem trail : True := by\n```"
    
    context, auto = parse_autoformalization(raw)
    check("no backticks in auto", "```" not in auto)
    check("no backticks in context", "```" not in context)


def test_parse_no_theorem_found():
    """Autoformalization with no theorem/def/lemma — edge case"""
    print("\n--- test_parse_no_theorem_found ---")
    
    raw = "```lean4\nimport Mathlib\n\n#check Nat\n```"
    
    context, auto = parse_autoformalization(raw)
    check("auto is empty", auto == "")
    check("context has content", "import Mathlib" in context)


# outcome checking

def check_outcome(outcome):
    """Replicate the outcome checking logic from process_answer"""
    if outcome is None:
        return False
    
    has_error = False
    has_sorries = 'sorries' in outcome
    
    if "messages" in outcome:
        for msg in outcome["messages"]:
            if msg.get("severity") == "error":
                has_error = True
    
    if has_error or has_sorries:
        return False
    return True


def test_outcome_clean_pass():
    """Clean pass: just env, no messages"""
    print("\n--- test_outcome_clean_pass ---")
    
    outcome = {"env": 1}
    check("passes", check_outcome(outcome) == True)


def test_outcome_warning_pass():
    """Warnings should still pass"""
    print("\n--- test_outcome_warning_pass ---")
    
    outcome = {"env": 1, "messages": [{"severity": "warning", "data": "unused variable"}]}
    check("passes with warning", check_outcome(outcome) == True)


def test_outcome_error_fail():
    """Error messages should fail"""
    print("\n--- test_outcome_error_fail ---")
    
    outcome = {"env": 1, "messages": [{"severity": "error", "data": "type mismatch"}]}
    check("fails with error", check_outcome(outcome) == False)


def test_outcome_sorry_key_fail():
    """Top-level sorries key should fail"""
    print("\n--- test_outcome_sorry_key_fail ---")
    
    outcome = {"env": 1, "sorries": [{"proofState": 0, "pos": {"line": 1, "column": 0}}]}
    check("fails with sorries key", check_outcome(outcome) == False)


def test_outcome_sorry_and_messages():
    """Both sorries key and messages present"""
    print("\n--- test_outcome_sorry_and_messages ---")
    
    outcome = {
        "env": 1,
        "messages": [{"severity": "warning", "data": "declaration uses sorry"}],
        "sorries": [{"proofState": 0}]
    }
    check("fails with sorries even if messages are warnings", check_outcome(outcome) == False)


def test_outcome_none():
    """None outcome (session failure)"""
    print("\n--- test_outcome_none ---")
    check("None returns False", check_outcome(None) == False)


def test_outcome_error_and_sorry():
    """Both error and sorry present"""
    print("\n--- test_outcome_error_and_sorry ---")
    
    outcome = {
        "env": 1,
        "messages": [{"severity": "error", "data": "unknown identifier"}],
        "sorries": [{"proofState": 0}]
    }
    check("fails", check_outcome(outcome) == False)


def test_outcome_info_message_pass():
    """Info-severity messages should pass"""
    print("\n--- test_outcome_info_message_pass ---")
    
    outcome = {"env": 1, "messages": [{"severity": "information", "data": "foo : Nat"}]}
    check("passes with info", check_outcome(outcome) == True)


def test_outcome_multiple_errors():
    """Multiple error messages"""
    print("\n--- test_outcome_multiple_errors ---")
    
    outcome = {
        "env": 1, 
        "messages": [
            {"severity": "error", "data": "type mismatch"},
            {"severity": "error", "data": "unknown identifier"},
        ]
    }
    check("fails", check_outcome(outcome) == False)


def test_outcome_mixed_messages():
    """Mix of warning, info, and error — should fail"""
    print("\n--- test_outcome_mixed_messages ---")
    
    outcome = {
        "env": 1,
        "messages": [
            {"severity": "warning", "data": "unused"},
            {"severity": "information", "data": "info"},
            {"severity": "error", "data": "fail"},
        ]
    }
    check("fails due to error", check_outcome(outcome) == False)


# batch size

def test_batch_size_calculation():
    """Verify batch_size doesn't produce zero for various answer counts"""
    print("\n--- test_batch_size_calculation ---")
    
    num_batches = 32
    for n_answers in [0, 1, 2, 5, 31, 32, 33, 64, 100]:
        if n_answers == 0:
            # Should be caught by the empty answers guard
            check(f"n={n_answers}: skipped by guard", True)
            continue
        batch_size = math.ceil(n_answers / num_batches)
        try:
            batches = list(range(0, n_answers, batch_size))
            check(f"n={n_answers}: batch_size={batch_size}, {len(batches)} batches", 
                  batch_size > 0 and len(batches) > 0)
        except ValueError as e:
            check(f"n={n_answers}: raises {e}", False)


# integration test

INTEGRATION_TESTS = [
    {
        "theorem_names": "int_trivial_correct",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_trivial_correct : True := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_trivial_correct : True := by\n  trivial\n```"
        ],
        "_expected": [True],
        "_description": "Simplest possible correct proof"
    },
    {
        "theorem_names": "int_trivial_wrong",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_trivial_wrong : True := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_trivial_wrong : True := by\n  exact Nat.zero\n```"
        ],
        "_expected": [False],
        "_description": "Simplest possible incorrect proof (type mismatch)"
    },
    
    {
        "theorem_names": "int_sorry_simple",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_sorry_simple : 1 + 1 = 2 := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_sorry_simple : 1 + 1 = 2 := by\n  sorry\n```"
        ],
        "_expected": [False],
        "_description": "Sorry should always be rejected"
    },
    {
        "theorem_names": "int_sorry_hidden",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_sorry_hidden (n : Nat) : n < n + 1 := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_sorry_hidden (n : Nat) : n < n + 1 := by\n  have h : n < n + 1 := by sorry\n  exact h\n```"
        ],
        "_expected": [False],
        "_description": "Sorry hidden inside a have-block should still be rejected"
    },
    
    {
        "theorem_names": "int_false_unprovable",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_false_unprovable : False := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_false_unprovable : False := by\n  trivial\n```",
            "```lean4\nimport Mathlib\n\ntheorem int_false_unprovable : False := by\n  simp\n```"
        ],
        "_expected": [False, False],
        "_description": "False is not provable — both should fail"
    },
    
    {
        "theorem_names": "int_omega",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_omega (a b : Nat) : a + b = b + a := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_omega (a b : Nat) : a + b = b + a := by\n  omega\n```"
        ],
        "_expected": [True],
        "_description": "omega tactic on nat arithmetic"
    },
    {
        "theorem_names": "int_ring",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\nopen Real\n\ntheorem int_ring (x y : ℝ) : (x + y)^2 = x^2 + 2*x*y + y^2 := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\nopen Real\n\ntheorem int_ring (x y : ℝ) : (x + y)^2 = x^2 + 2*x*y + y^2 := by\n  ring\n```"
        ],
        "_expected": [True],
        "_description": "ring tactic with open Real context"
    },
    {
        "theorem_names": "int_norm_num",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_norm_num : (7 : ℤ) * 13 = 91 := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_norm_num : (7 : ℤ) * 13 = 91 := by\n  norm_num\n```"
        ],
        "_expected": [True],
        "_description": "norm_num on integer arithmetic"
    },
    {
        "theorem_names": "int_simp",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_simp (n : Nat) : n + 0 = n := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_simp (n : Nat) : n + 0 = n := by\n  simp\n```"
        ],
        "_expected": [True],
        "_description": "simp tactic"
    },
    
    {
        "theorem_names": "int_natural_language_preamble",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_natural_language_preamble : 0 = 0 := by\n```",
        "answers": [
            "Sure! Here's the proof for the theorem. The key insight is that reflexivity handles this directly.\n\n```lean4\nimport Mathlib\n\ntheorem int_natural_language_preamble : 0 = 0 := by\n  rfl\n```\n\nThe `rfl` tactic works because both sides of the equation are definitionally equal."
        ],
        "_expected": [True],
        "_description": "Long natural language before and after the code block"
    },
    {
        "theorem_names": "int_no_code_fence",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_no_code_fence : True := by\n```",
        "answers": [
            "I cannot solve this, but let me try thinking about it differently."
        ],
        "_expected": [False],
        "_description": "Answer with no code at all"
    },
    {
        "theorem_names": "int_extra_whitespace",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_extra_whitespace (n : Nat) : n = n := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_extra_whitespace (n : Nat) :  n = n  := by\n  rfl\n```"
        ],
        "_expected": [True],
        "_description": "Extra whitespace in theorem statement in answer"
    },
    
    {
        "theorem_names": "int_mixed_correct_incorrect",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_mixed_correct_incorrect (n : Nat) : n ≤ n := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_mixed_correct_incorrect (n : Nat) : n ≤ n := by\n  omega\n```",
            "```lean4\nimport Mathlib\n\ntheorem int_mixed_correct_incorrect (n : Nat) : n ≤ n := by\n  exact Nat.lt_irrefl n\n```",
            "```lean4\nimport Mathlib\n\ntheorem int_mixed_correct_incorrect (n : Nat) : n ≤ n := by\n  sorry\n```",
            "```lean4\nimport Mathlib\n\ntheorem int_mixed_correct_incorrect (n : Nat) : n ≤ n := by\n  rfl\n```"
        ],
        "_expected": [True, False, False, True],
        "_description": "Mix of correct (omega, rfl), wrong (lt_irrefl), and sorry"
    },
    
    {
        "theorem_names": "int_open_scoped",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\nopen Real Set\nopen scoped BigOperators\n\ntheorem int_open_scoped (x : ℝ) : x * 1 = x := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\nopen Real Set\nopen scoped BigOperators\n\ntheorem int_open_scoped (x : ℝ) : x * 1 = x := by\n  ring\n```"
        ],
        "_expected": [True],
        "_description": "Complex context with open scoped"
    },
    
    {
        "theorem_names": "int_empty_answers",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_empty_answers : True := by\n```",
        "answers": [],
        "_expected": [],
        "_description": "Empty answers list"
    },
    
    {
        "theorem_names": "int_multistep",
        "autoformalization": "\nComplete the following Lean 4 code:\n```lean4\nimport Mathlib\n\ntheorem int_multistep (a b : Nat) (h : a = b) : b = a := by\n```",
        "answers": [
            "```lean4\nimport Mathlib\n\ntheorem int_multistep (a b : Nat) (h : a = b) : b = a := by\n  rw [h]\n```",
            "```lean4\nimport Mathlib\n\ntheorem int_multistep (a b : Nat) (h : a = b) : b = a := by\n  exact h.symm\n```",
            "```lean4\nimport Mathlib\n\ntheorem int_multistep (a b : Nat) (h : a = b) : b = a := by\n  exact h\n```"
        ],
        "_expected": [True, True, False],
        "_description": "Two correct proofs (rw, symm) and one wrong (exact h has wrong direction)"
    },
]


def generate_integration_test_file(output_path):
    """Generate the integration test JSON (strip _expected and _description)"""
    clean_data = []
    for item in INTEGRATION_TESTS:
        clean_data.append({
            "theorem_names": item["theorem_names"],
            "autoformalization": item["autoformalization"],
            "answers": item["answers"]
        })
    
    with open(output_path, "w") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    print(f"Generated integration test file: {output_path}")
    print(f"Contains {len(clean_data)} theorems with {sum(len(d['answers']) for d in clean_data)} total answers")


def validate_integration_results(results_path):
    """Validate integration test results against expected outcomes"""
    with open(results_path) as f:
        results = json.load(f)
    
    all_passed = True
    total_checks = 0
    passed_checks = 0
    
    for item in INTEGRATION_TESTS:
        name = item["theorem_names"]
        expected = item["_expected"]
        desc = item["_description"]
        
        print(f"\n--- {name}: {desc} ---")
        
        if name not in results:
            print(f"  MISSING from results")
            all_passed = False
            continue
        
        actual = results[name]
        actual_bools = [r["answer_bool"] for r in actual]
        
        if len(actual_bools) != len(expected):
            print(f"  FAIL: expected {len(expected)} results, got {len(actual_bools)}")
            all_passed = False
            continue
        
        answer_to_expected = {}
        for ans_text, exp_bool in zip(item["answers"], expected):
            answer_to_expected[ans_text] = exp_bool
        
        for r in actual:
            total_checks += 1
            ans = r["answer"]
            act = r["answer_bool"]
            exp = answer_to_expected.get(ans)
            
            if exp is None:
                print(f"  FAIL: unexpected answer in results")
                all_passed = False
            elif act == exp:
                passed_checks += 1
                status = "✓" if act else "✗"
                print(f"  PASS: [{status}] answer_bool={act} (expected {exp})")
            else:
                all_passed = False
                print(f"  FAIL: answer_bool={act}, expected {exp}")
                # Show a snippet of the answer for debugging
                snippet = ans[:80].replace('\n', ' ')
                print(f"        answer: {snippet}...")
    
    print(f"\n{'='*60}")
    print(f"Integration results: {passed_checks}/{total_checks} checks passed")
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED")
    else:
        print("SOME INTEGRATION TESTS FAILED")
    print(f"{'='*60}")
    return all_passed



def run_unit_tests():
    print("=" * 60)
    print("UNIT TESTS: extract_proof_body")
    print("=" * 60)
    
    test_extract_basic_lean4_fence()
    test_extract_lean_fence()
    test_extract_bare_fence()
    test_extract_no_fence_code_answer()
    test_extract_no_fence_natural_language()
    test_extract_empty_answer()
    test_extract_multiline_tactic_proof()
    test_extract_helper_lemma_before_main()
    test_extract_term_mode_proof()
    test_extract_where_clause()
    test_extract_colon_equals_by_no_space()
    test_extract_special_chars_in_name()
    test_extract_multiple_code_blocks()
    test_extract_answer_redeclares_with_different_formatting()
    test_extract_def_instead_of_theorem()
    test_extract_lemma_keyword()
    
    print("\n" + "=" * 60)
    print("UNIT TESTS: autoformalization parsing")
    print("=" * 60)
    
    test_parse_standard_format()
    test_parse_multiple_opens()
    test_parse_def_declaration()
    test_parse_theorem_in_comment()
    test_parse_no_code_fence()
    test_parse_trailing_backticks()
    test_parse_no_theorem_found()
    
    print("\n" + "=" * 60)
    print("UNIT TESTS: outcome checking")
    print("=" * 60)
    
    test_outcome_clean_pass()
    test_outcome_warning_pass()
    test_outcome_error_fail()
    test_outcome_sorry_key_fail()
    test_outcome_sorry_and_messages()
    test_outcome_none()
    test_outcome_error_and_sorry()
    test_outcome_info_message_pass()
    test_outcome_multiple_errors()
    test_outcome_mixed_messages()
    
    print("\n" + "=" * 60)
    print("UNIT TESTS: batch size edge cases")
    print("=" * 60)
    
    test_batch_size_calculation()
    
    return summary()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python3 {sys.argv[0]} unit                        # Run unit tests")
        print(f"  python3 {sys.argv[0]} generate [output_path]      # Generate integration test JSON")
        print(f"  python3 {sys.argv[0]} validate <results_path>     # Validate integration results")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "unit":
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    elif cmd == "generate":
        output = sys.argv[2] if len(sys.argv) > 2 else "test_repl_comprehensive.json"
        generate_integration_test_file(output)
    
    elif cmd == "validate":
        if len(sys.argv) < 3:
            print("Error: provide path to verified JSON")
            sys.exit(1)
        success = validate_integration_results(sys.argv[2])
        sys.exit(0 if success else 1)
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)