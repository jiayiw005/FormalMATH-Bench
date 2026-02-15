/-
This file was edited by Aristotle.

Lean Toolchain version: leanprover/lean4:v4.20.0-rc5
Mathlib version: d62eab0cc36ea522904895389c301cf8d844fd69 (May 9, 2025)
-/

import Mathlib.Data.Nat.Basic
import Mathlib.Tactic


/-!
Simple test theorems for Aristotle
-/

/- Aristotle failed to load this code into its environment. Double check that the syntax is correct.

'zero_add' has already been declared-/
-- Basic arithmetic theorems
theorem zero_add (n : ℕ) : 0 + n = n := by
  sorry