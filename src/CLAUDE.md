# [VIP-vision-PDDL]

[One-line description: learns representation learning for automated planning from noisy-classified images]

## Stack

- Python 3.11+

## Code style

- Type hints on all function signatures and return types. Strict mode — no implicit `Any`.
- Docstrings on public functions and classes (Google style). Skip for trivially named private helpers.
- No module-level mutable state. Pass dependencies as arguments or via a context object.
- Prefer functions over classes unless you genuinely need state or polymorphism.
- Keep functions short. If a function needs section-comment headers to be readable, split it.

## What to do / not do

- ✅ When you're choosing between a clever one-liner and an obvious five-liner, pick the five-liner.
- ✅ Flag architectural decisions explicitly when you make them (which library, which pattern, why) — don't bury them in code.
- ❌ Don't add abstractions for hypothetical future needs. Add them when the second concrete use case appears.
- ❌ Don't add dependencies without asking.
