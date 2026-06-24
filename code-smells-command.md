Analyze the file at `$ARGUMENTS` for code smells. Read the entire file first, then work through it systematically checking each category below.

**Verification rule:** Before flagging something as dead code or unused, Grep the project to confirm. False positives undermine trust — only flag what you've verified.

## Categories to check

1. **Dead code** — functions, classes, methods, or variables defined but never referenced anywhere in the project.
2. **Unused imports** — modules imported but never used in the file.
3. **Redundant imports** — the same module imported at the top and again inside a function body.
4. **Hardcoded values** — absolute paths, magic numbers, credentials, or environment-specific strings that should come from config, `__file__`, CLI args, or env vars.
5. **Module-level side effects** — code that runs on import (prints, network calls, file I/O) outside of `if __name__ == "__main__"`.
6. **Dead variables** — variables assigned but never read afterward.
7. **God functions/classes** — functions with too many parameters (>7), excessive length (>80 lines), or too many responsibilities.
8. **Misleading names** — variable or parameter names that no longer match what they represent after a refactor.
9. **Inconsistency** — the same concept controlled two different ways (e.g., global constant AND CLI arg), or naming conventions that break the file's own patterns.
10. **Commented-out code** — large blocks of commented-out alternatives left without explanation.
11. **Bare or overly broad exception handling** — `except: pass`, `except Exception`, or swallowed errors violating project conventions.
12. **Duplicated logic** — the same block of code appearing in multiple places within the file.
13. **Identity mappings or no-op logic** — dicts that map keys to themselves, conditionals that always take the same branch, etc.

## Output format

A numbered list. Each item has:
- **Line number(s)**
- **Category** (short label)
- **What's wrong** — one or two sentences with the actual symbol names/values
- **Suggestion** — brief fix (when non-obvious)

Skip categories where the file is clean. Don't review for correctness/bugs, don't suggest architectural redesigns, don't flag pure style preferences unless they violate CLAUDE.md conventions.
