hiking_object_detection_prompt = """
You are a visual object-recognition agent for a robotic planning system.

The image shows an N×M grid (e.g., 3×3, 4x5).
Each cell is of type loc (location).

Your task is to:
1. Count how many cells are present in the image.
2. Detect the grid size NxM (number of columns/rows).
3. Create position objects rI_cJ for each cell in the grid.
4. Output the objects in a format suitable for the following domain:

-----------------------------------------------------
OBJECT TYPES
-----------------------------------------------------
You must create objects of these types:

- loc  → a grid cell (e.g., r0_c1, r1_c1, ..., rN_cM)
This the only type you are allowed to output.
-----------------------------------------------------
Position naming:
- Let the grid be N×M.
- Define one position object for each cell: rI_cJ
  where:
    I ∈ {0..N-1} is the row index (y-coordinate, top to bottom)
    J ∈ {0..M-1} is the column index (x-coordinate, left to right)
- The top-left cell is r0_c0.
- The cell to its right is r0_c1, etc.
- The cell directly below r0_c0 is r1_c0, etc.

-----------------------------------------------------
EXAMPLE OUTPUT
-----------------------------------------------------

r0_c1:loc
r1_c3:loc
"""


def confidence_system_prompt(location_names):
    locations = ", ".join(location_names)

    return f"""
You are a visual reasoning agent for a robotic planning system.

KNOWN OBJECTS (complete lists, do NOT add or remove):
- Locations (grid cells): {locations} (type=loc)

IMPORTANT CONSTRAINTS ON OBJECTS:
- Locations are named rI_cJ where I = row (top→bottom), J = column (left→right), i.e., r0_c0 is the top-left cell,
    r0_c1 is the cell to its right, r1_c0 is the cell below r0_c0, etc.

-----------------------------------------------------
VISUAL DEFINITIONS (IMPORTANT)
-----------------------------------------------------
- BROWN cell: brown tile.
- BLUE cell: blue/turquoise tile.
- GREEN cell: green tile.
- WHITE cell: white/bright tile.
- PERSON cell: the person sprite.
- STAR cell: Golden 5-star.

=====================================================
BLOCK 1 — PERCEPTION (MANDATORY)
=====================================================
For EVERY location loc-I-J output EXACTLY one line:

    cell(loc-I-J:location) = {{brown | blue | green | white | person | star}}

1. For every cell describe the object briefly:
   - (row, col): short description of colors and shape.
2. From those descriptions, decide which cell contains the PERSON
   and which cell contains the STAR.
3. Hold the location of the PERSON in memory to output later, in this json format:
 {{"person_at": [row, col]}}
 
 
=====================================================
BLOCK 2 — PREDICATES WITH SCORES
=====================================================

(1) Person and goal:
- at(x:loc): 2   iff   the center of the person sprite is at location x, i.e. cell(x)=person.
  (Do NOT output the 0-score at() lines; omitted means score 0.)
- is_goal(x:loc): 2   iff   cell(x)=star.

(2) Neighboring cells:
Let FROM = rI1_cJ1, TO = rI2-cJ2.
- adjacent(FROM:loc,TO:loc)  2  iff
    I2 = I1 - 1  AND J2 = J1  OR
    I2 = I1 + 1  AND J2 = J1  OR
    I2 = I1      AND J2 = J1 - 1  OR
    I2 = I1      AND J2 = J1 + 1 

-----------------------------------------------------
CONFIDENCE SCORES
-----------------------------------------------------
- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → The predicate MIGHT hold, but evidence is unclear, partial, or occluded.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.


-----------------------------------------------------
OUTPUT FORMAT (STRICT)
-----------------------------------------------------
Output only predicates of format of BLOCK 2.
One line per item.
Format in BLOCK 2:
    <predicate>: <score>
No explanations.
"""
