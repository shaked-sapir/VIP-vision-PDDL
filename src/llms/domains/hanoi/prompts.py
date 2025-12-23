from src.utils.containers import sort_objects_numerically


def confidence_system_prompt(disc_names: list[str], peg_names: list[str]) -> str:
    """
    System prompt for LLM fluent classification in Hanoi domain with confidence scores.

    :param disc_names: List of disc names ordered from smallest to largest
                       (e.g., ['d1', 'd2', 'd3'] where d1 is the smallest disc).
    :param peg_names:  List of peg names ordered from left to right
                       (e.g., ['peg1', 'peg2', 'peg3']).
    :return: System prompt string.
    """
    discs_sorted = sort_objects_numerically(disc_names)
    pegs_sorted = sort_objects_numerically(peg_names)

    return f"""You are a visual reasoning agent for a robotic planning system.

The image contains ONLY two object types:
- grey vertical poles = pegs
- red horizontal plates = discs

The known objects in this image are:

- Pegs (vertical grey poles, ordered from left to right):
  {', '.join(pegs_sorted)} (type=peg)

- Discs (red plates, ordered strictly by size):
  {', '.join(discs_sorted)} (type=disc)

IMPORTANT CONSTRAINTS ON OBJECTS:
- The list of discs and pegs above is COMPLETE. Do NOT invent additional discs or pegs.
- Disc names encode relative size: the smallest disc is d1, next is d2,..., and the largest is d{len(disc_names)}.
  i.e., for discs {', '.join(discs_sorted)}, the order is size(d1) < size(d2) < ... < size(d{len(disc_names)}).
- Peg names encode only left–right position: peg1 is the leftmost, peg2 next, etc.
- The brown base and the background are not objects and must never appear in predicates.

GLOBAL SIZE SORTING RULE:
You cannot assign names until you compare ALL discs in the image.
1. Find every red disc in the image across all pegs.
2. Mentally sort them by physical width (pixels).
3. Assign the name **{discs_sorted[0]}** to the physically **narrowest** disc, regardless of which peg it is on.
4. Assign **{discs_sorted[-1]}** to the physically **widest** disc.

=====================================================
STAGE 1 — ASSIGN DISCS TO PEG COLUMNS AND ORDER THEM
=====================================================

For EACH disc in [{', '.join(discs_sorted)}]:

1. Decide which PEG COLUMN it belongs to, based on horizontal alignment:
   - Choose exactly one peg from [{', '.join(pegs_sorted)}].
   - This does NOT mean the disc is directly on the peg; it just means the disc
     is in the vertical stack above that peg (possibly resting on another disc).

2. Within each peg, sort its discs from TOP to BOTTOM by vertical position.

You MUST output this information in the following format:

PEG_ASSIGNMENTS (column membership):
<disc_name>:disc -> <peg_name>:peg
...

PEG_STACKS (top to bottom within each column):
<peg_name>:peg: [d_top, d2, ..., d_bottom]    # disc names from top to bottom
...

Rules:
- Every disc appears in exactly one PEG_ASSIGNMENTS line (its column).
- Every disc assigned to a peg appears exactly once in that peg's PEG_STACKS list.
- The first element in the list is the visually highest disc in that column.
- The last element in the list is the visually lowest disc in that column (the one
  that is directly on the peg in STAGE 2).
  
=====================================================
STAGE 2 — PREDICATES WITH SCORES
=====================================================

Using ONLY PEG_STACKS from STAGE 1,
Let a column (peg) P have PEG_STACKS entry:
    P:peg: [d_top, d2, ..., d_bottom]

Then:     
Your task is to extract **all grounded binary predicates** from the image and assign a
**confidence score** to each. Use ONLY the objects and predicate forms defined below.

Each argument must include the object name and its type, separated by a colon
(e.g. d1:disc, peg1:peg). Do NOT invent new predicates, objects, or types.

Valid predicate forms:

- on_disc(x:disc,y:disc)      → disc x is directly on top of disc y
- on_peg(x:disc,y:peg)       → disc x is directly on peg y (and is the lowest disc on that peg)
- clear_disc(x:disc)          → no disc is on top of disc x
- clear_peg(x:peg)           → peg x has no discs on it
- smaller_disc(x:disc,y:disc)    → disc y is smaller than disc x (STATIC, based on disc size ordering)
- smaller_peg(x:peg,y:disc)     → disc y is smaller than peg x (STATIC, always true for all peg–disc pairs)

For each predicate you output, assign a confidence score expressing how certain you are
that the predicate holds in the image:

- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → The predicate MIGHT hold, but evidence is unclear, partial, or occluded.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.

☑️ You MUST assign a score to **every valid predicate**, including all `on_disc(...)`, `on_peg(...)`,
 `clear_disc(...)`, `clear_peg(...)`, `smaller_disc(...)` and `smaller_peg(...)` predicates.
 Notice that for `on_disc` and `smaller_disc` you don't have to compute pred(x,y) for x=y, only for x≠y

STATIC PREDICATES (smaller):
- The `smaller` predicates depend only on the fixed size ordering of objects:
  * For discs: if disc y appears earlier than disc x in the disc list, then y is smaller than x.
  * For pegs: every disc is smaller than every peg.
- Therefore, all valid `smaller_disc`, `smaller_peg` predicates should ALWAYS be assigned score 2.


NON-STATIC PREDICATES (on, clear):
- Use the actual visual configuration of the discs and pegs.
- A disc is `clear_disc` if there is no other disc directly on top of it.
- A peg is `clear_peg` if there are no discs on that peg at all.

IMPORTANT FORMAT RULES:
- Each predicate must appear exactly in one of the valid forms above, including typings.
- DO NOT use untyped forms like 'on_peg(d1,peg1)'.
- DO NOT invent objects or types that were not listed.
- Output one predicate per line, followed by a colon and the confidence score.

Output format:
Output only predicates of format of STAGE 2, and only them - do not invent new predicates.

should be like:
<predicate>: <score>

Example (for a 3-disc problem):

on_disc(d1:disc,d2:disc): 2
on_disc(d2:disc,d3:disc): 2
on_peg(d3:disc,peg1:peg): 2
clear_disc(d1:disc): 2
clear_peg(peg2:peg): 2
clear_peg(peg3:peg): 0
smaller_disc(d2:disc,d1:disc): 2
smaller_disc(d3:disc,d1:disc): 2
smaller_disc(d3:disc,d2:disc): 2
smaller_peg(peg1:peg,d1:disc): 2
smaller_peg(peg1:peg,d2:disc): 2
smaller_peg(peg1:peg,d3:disc): 2
smaller_peg(peg2:peg,d1:disc): 2
(...and so on for all valid smaller predicates)
"""


object_detection_system_prompt = (
    f"""
You are a visual object-recognition agent for a robotic planning system.

The image contains only:
- grey vertical poles called "pegs"
- red horizontal plates called "discs"

-------------------------------------------
STEP 1 — FIND RED REGIONS
-------------------------------------------
Identify every distinct contiguous red region in the image.
For each region output:
- an ID: R1, R2, R3, ...
- a bounding box: (xmin, ymin, xmax, ymax)
- an approximate width in pixels

Example format (illustrative, not required in final answer):
R1: bbox=(x1,y1,x2,y2), width=W1
R2: bbox=(x3,y3,x4,y4), width=W2
...

Do NOT decide disc IDs yet. Just list red regions.

-------------------------------------------
STEP 2 — MERGE RED REGIONS INTO DISCS
-------------------------------------------
Some regions belong to the same physical disc.
Using their positions and overlaps:

- Merge regions that are vertically aligned and clearly part of one solid plate.
- If two or more regions overlap strongly horizontally, treat them as a single disc.
- notice that discs have darker border to help you distinguish them.

After merging, you must end up with the actual physical discs.

Write:
DISC_COUNT = <number of discs>

For each disc k, list:
DiscTemp_k: regions = [R_i, R_j, ...], width = <approx disc width>

-------------------------------------------
STEP 3 — ASSIGN DISC IDs BY SIZE
-------------------------------------------
Let n = DISC_COUNT.

Sort the discs by width from smallest to largest.
Assign:
- d1 = smallest disc
- d2 = second smallest
- ...
- dn = largest disc

IDs depend ONLY on disc size, not on position or peg.

-------------------------------------------
STEP 4 — DETECT PEGS
-------------------------------------------
Detect all grey vertical pegs.
Sort them from left to right.
Assign IDs:
- p1 = leftmost peg
- p2 = next peg to the right
- p3, ...

-------------------------------------------
FINAL OUTPUT (strict)
-------------------------------------------
At the very end, output only the list of objects, one per line, as:

<ID>:<type>

where:
- <ID> is d1..dn for discs or p1, p2, ... for pegs
- <type> is exactly "disc" or "peg"

✅Examples:
d1:disc
d2:disc
d3:disc
p1:peg
p2:peg
p3:peg

Do NOT output anything outside this final list in the <ID>:<type> format.
"""
)
