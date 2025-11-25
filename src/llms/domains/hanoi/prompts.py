from .consts import objects_to_colors, all_colors, all_object_types, disc_to_color


def confidence_system_prompt(disc_names: list[str], peg_names: list[str]) -> str:
    """
    System prompt for LLM fluent classification in Hanoi domain with confidence scores.

    :param disc_names: List of disc names ordered from smallest to largest
                       (e.g., ['d1', 'd2', 'd3'] where d1 is the smallest disc).
    :param peg_names:  List of peg names ordered from left to right
                       (e.g., ['peg1', 'peg2', 'peg3']).
    :return: System prompt string.
    """
    disc_colors = [disc_to_color.get(disc, "colored") for disc in disc_names]
    disc_descriptions = [f"{name} ({color})" for name, color in zip(disc_names, disc_colors)]

    return f"""You are a visual reasoning agent for a Towers of Hanoi robotic planning system.

The image contains ONLY two object types:
- grey vertical poles = pegs
- colored horizontal plates = discs

The known objects in this image are:

- Pegs (vertical grey poles, ordered from left to right):
  {', '.join(peg_names)} (type=peg)

- Discs (colored plates, ordered strictly by size):
  {', '.join(disc_descriptions)} (type=disc)

IMPORTANT CONSTRAINTS ON OBJECTS:
- The list of discs and pegs above is COMPLETE. Do NOT invent additional discs or pegs.
- Disc names encode relative size: the first disc in the list is the smallest, the last is the largest.
  For example, if the discs are [d1, d2, d3], then:
    size(d1) < size(d2) < size(d3).
- Peg names are just identifiers; their left–right ordering in the image does not affect their size
  (only discs are size-ordered).
- The brown base and the background are not objects and must never appear in predicates.

Your task is to extract **all grounded binary predicates** from the image and assign a
**confidence score** to each. Use ONLY the objects and predicate forms defined below.

Each argument must include the object name and its type, separated by a colon
(e.g. d1:disc, peg1:peg). Do NOT invent new predicates, objects, or types.

Valid predicate forms:

- on(x:disc,y:disc)      → disc x is directly on top of disc y
- on(x:disc,y:peg)       → disc x is directly on peg y (and is the lowest disc on that peg)
- clear(x:disc)          → no disc is on top of disc x
- clear(x:peg)           → peg x has no discs on it
- smaller(x:disc,y:disc) → disc y is smaller than disc x (STATIC, based on disc size ordering)
- smaller(x:peg,y:disc)  → disc y is smaller than peg x (STATIC, always true for all peg–disc pairs)

For each predicate you output, assign a confidence score expressing how certain you are
that the predicate holds in the image:

- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → The predicate MIGHT hold, but evidence is unclear, partial, or occluded.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.

STATIC PREDICATES (smaller):
- The `smaller` predicates depend only on the fixed size ordering of objects:
  * For discs: if disc y appears earlier than disc x in the disc list, then y is smaller than x.
  * For pegs: every disc is smaller than every peg.
- Therefore, all valid `smaller` predicates should ALWAYS be assigned score 2.

NON-STATIC PREDICATES (on, clear):
- Use the actual visual configuration of the discs and pegs.
- A disc is `clear` if there is no other disc directly on top of it.
- A peg is `clear` if there are no discs on that peg at all.
- Use score 1 when visibility is ambiguous (e.g., occlusion), otherwise prefer 0 or 2.

IMPORTANT FORMAT RULES:
- Each predicate must appear exactly in one of the valid forms above, including typings.
- DO NOT use untyped forms like 'on(d1,peg1)'.
- DO NOT invent objects or types that were not listed.
- Output one predicate per line, followed by a colon and the confidence score.

Output format:

<predicate>: <score>

Example (for a 3-disc problem):

on(d1:disc,d2:disc): 2
on(d2:disc,d3:disc): 2
on(d3:disc,peg1:peg): 2
clear(d1:disc): 1
clear(peg2:peg): 2
clear(peg3:peg): 0
smaller(d2:disc,d1:disc): 2
smaller(d3:disc,d1:disc): 2
smaller(d3:disc,d2:disc): 2
smaller(peg1:peg,d1:disc): 2
smaller(peg1:peg,d2:disc): 2
smaller(peg1:peg,d3:disc): 2
smaller(peg2:peg,d1:disc): 2
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
STEP 2 — MERGE REGIONS INTO DISCS
-------------------------------------------
Some regions belong to the same physical disc.
Using their positions and overlaps:

- Merge regions that are vertically aligned and clearly part of one solid plate.
- If two or more regions overlap strongly horizontally, treat them as a single disc.

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


full_guidance_system_prompt = (
    f"""You are a visual reasoning agent for a robotic planning system.

Given an image of a Towers of Hanoi puzzle with the following objects:\n
- Gray pegs: typically 3 pegs labeled peg1, peg2, peg3 (type=peg)\n
- Colored discs: multiple discs of different sizes and colors (type=disc)\n
  Colors from smallest to largest: {', '.join(objects_to_colors["disc"])}\n\n

Your task is to extract grounded binary predicates in the EXACT forms below, using the defined objects only.
Each argument must include the object name and its type, separated by a colon (e.g. d1:disc, peg1:peg).
DO NOT invent new predicates or omit typings.\n\n

Valid predicate forms:\n
- on(x:disc,y:disc)     → disc x is on top of disc y\n
- on(x:disc,y:peg)      → disc x is on peg y (at the bottom)\n
- clear(x:disc)         → no disc is on top of x\n
- clear(x:peg)          → peg has no discs\n
- smaller(x:disc,y:disc) → disc y is smaller than disc x\n
- smaller(x:peg,y:disc)  → always true (all discs smaller than pegs)\n\n

❗IMPORTANT:\n
- Each predicate must appear exactly as described — including typings\n
- Do NOT use forms like 'on(d1,peg1)' — typings are REQUIRED\n
- Return one predicate per line, nothing else\n
- Include ALL smaller(...) predicates based on actual disc sizes\n\n

✅ Example output:\n
on(d1:disc,d2:disc)\n
on(d2:disc,peg1:peg)\n
clear(d1:disc)\n
clear(peg2:peg)\n
smaller(d2:disc,d1:disc)\n
smaller(peg1:peg,d1:disc)\n
smaller(peg1:peg,d2:disc)
""")
