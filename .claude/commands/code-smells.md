---
name: code-smells
description: >
  Analyze a source code file for code smells — dead code, unused variables, redundant imports, 
  hardcoded values, god functions, misleading names, violation of project conventions, and more. 
  Use when: "code smells in X", "what's wrong with this file", "audit this file", "review X for 
  issues", "clean up X", or any request to find problems, messiness, or improvement opportunities 
  in a specific file. Also use when the user passes a file path and asks for a quality check or 
  review without specifying what kind.
---

# Code Smells Analyzer

Analyze a source code file and produce a numbered list of concrete code smells, each with exact line numbers and a short explanation.

## Input

The user provides a file path (or the file is already in context). Read the entire file before analysis.

## How to analyze

Work through the file systematically, checking for each category below. Use Grep to verify cross-references (e.g., whether a function is called elsewhere in the project, whether an import is used, whether a variable is read after being assigned).

### Categories to check

1. **Dead code** — functions, classes, methods, or variables defined but never referenced anywhere in the project. Grep the project for each suspicious symbol before flagging it.
2. **Unused imports** — modules imported but never used in the file.
3. **Redundant imports** — the same module imported both at the top of the file and again inside a function body.
4. **Hardcoded values** — absolute paths, magic numbers, credentials, or environment-specific strings that should come from config, `__file__`, CLI args, or environment variables.
5. **Module-level side effects** — code that runs on import (prints, network calls, file I/O) outside of `if __name__ == "__main__"`.
6. **Dead variables** — variables assigned but never read afterward (write-only).
7. **God functions/classes** — functions with too many parameters (>7), excessive length (>80 lines), or too many responsibilities.
8. **Misleading names** — variable or parameter names that no longer match what they represent (e.g., after a refactor renamed the concept but not all references).
9. **Inconsistency** — the same concept controlled two different ways (e.g., a value set both as a global constant and via CLI arg), or naming conventions that break the file's own patterns.
10. **Commented-out code** — large blocks of commented-out alternatives left in place without explanation.
11. **Bare or overly broad exception handling** — `except: pass`, `except Exception`, or swallowed errors that violate project conventions.
12. **Duplicated logic** — the same block of code appearing in multiple places within the file.
13. **Identity mappings or no-op logic** — dicts/functions that map most keys to themselves, conditionals that always take the same branch, etc.

## Output format

A numbered list. Each item has:
- **Line number(s)** — exact lines where the smell occurs
- **Category** — short label (e.g., "Dead code", "Hardcoded path")  
- **What's wrong** — one or two sentences explaining the problem and why it matters
- **Suggestion** — brief fix recommendation (optional, include when non-obvious)

Be concrete and specific — cite the actual symbol names, values, or patterns. Don't pad with generic advice. If the file is clean in a category, skip it silently.

## Verification

Before flagging something as dead code or unused, always Grep the project to confirm. A function that looks unused in one file might be imported elsewhere. False positives undermine trust — only flag what you've verified.

## What NOT to do

- Don't review the file for correctness, bugs, or feature suggestions — this is strictly about code smells and maintainability.
- Don't suggest architectural redesigns — keep suggestions scoped to the file.
- Don't flag stylistic preferences (single vs double quotes, blank line count) unless they violate the project's own stated conventions (e.g., in CLAUDE.md).