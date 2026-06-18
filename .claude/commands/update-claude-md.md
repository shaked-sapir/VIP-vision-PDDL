Update the project's CLAUDE.md to reflect the current state of the codebase. Follow these steps carefully:

## Step 1 — Read the current CLAUDE.md
Read `/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/CLAUDE.md` in full before doing anything else.

## Step 2 — Re-explore the codebase
Use the Explore agent or file tools to gather up-to-date information on:

1. **Module map**: List all subdirectories under `src/`. For each, identify its single responsibility based on the files inside it.

2. **Base classes**: Re-read the following files (if they still exist) and note any signature changes or new abstract methods:
   - `src/object_detection/base_object_detector.py`
   - `src/fluent_classification/base_fluent_classifier.py`
   - `src/trajectory_handlers/image_trajectory_handler.py`
   - `src/pi_sam/masking/masking_strategies.py`
   - Any new base classes added since (look for files named `base_*.py`)

3. **Utils inventory**: List all `.py` files in `src/utils/` and their public functions (scan the `def ` lines). Compare with the lookup table in CLAUDE.md.

4. **New patterns**: Scan `src/` for new mixins, new strategy classes, or new abstract base classes not yet documented in CLAUDE.md.

5. **Domain list**: Check `src/domains/`, `src/object_detection/`, `src/fluent_classification/`, and `src/trajectory_handlers/` for any new or removed domains.

## Step 3 — Identify what changed
Compare what you found with the current CLAUDE.md content. List the deltas explicitly before editing:
- New modules / removed modules
- New base classes or changed signatures
- New or removed utils functions
- New domains
- New architectural patterns

If nothing changed, report that and stop — do not edit the file unnecessarily.

## Step 4 — Update CLAUDE.md in place
Edit only the sections that are out of date. Preserve everything else verbatim (especially Coding Standards, What Not To Do, and the project overview unless the architecture genuinely changed).

Sections that are likely to drift and should always be re-checked:
- **Module Map** table
- **Base Class Contracts** (method signatures)
- **`src/utils/` lookup table** inside the Modularity section
- **Adding a New Domain** checklist
- **Key Architectural Patterns** (if new patterns were introduced)

Keep the same format, heading structure, and tone as the original file.

## Step 5 — Report the changes
After editing, summarize in 3–5 bullet points exactly what was updated and why. If sections were left unchanged, say so.
