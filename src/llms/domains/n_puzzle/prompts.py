from src.utils.containers import sort_objects_numerically

npuzzle_object_detection_prompt = """
You are a visual object-recognition agent for a robotic planning system.

The image shows an N×N grid (e.g., 3×3, 4×4, 5×5).
Each non-empty cell contains exactly one tile with a visible integer number.
Exactly one cell is blank; the blank is NOT a tile.

Your task is to:
1. Count how many numbered tiles are present in the image.
2. Detect the grid size N (number of columns/rows).
3. Create position objects p_I_J for each cell in the grid.
4. Create tile objects t_K for each numbered tile (K is the number printed on the tile).
5. For each tile, determine which position it occupies.
6. Determine which position is blank (empty).
7. Output the objects and their locations in a format suitable for the following domain:

-----------------------------------------------------
OBJECT TYPES
-----------------------------------------------------
You must create objects of these types:

- tile      → a numbered square (e.g., t_1, t_2, t_3, ...)
- position  → a grid cell (e.g., p1_1, p1_2, ..., p_N_N)
These are the only two types you are allowed to output.
-----------------------------------------------------
Position naming:
- Let the grid be N×N.
- Define one position object for each cell: p_I_J
  where:
    I ∈ {1..N} is the column index (x-coordinate, left to right)
    J ∈ {1..N} is the row index (y-coordinate, top to bottom)
- The top-left cell is p_1_1.
- The cell to its right is p_2_1, etc.
- The cell directly below p_1_1 is p_1_2, etc.

Tile naming:
- Every cell containing a visible integer number represents exactly one tile.
- If a tile shows number K, its object name MUST be t_K (type tile).
  For example:
    tile with number 7 → t_7:tile

-----------------------------------------------------
EXAMPLE OUTPUT
-----------------------------------------------------
t_1:tile
t_2:tile
t_3:tile
p_1_2:position
p_2_2:position
"""


def confidence_system_prompt(tile_names: list[str], position_names: list[str]):
    tiles = ", ".join(sort_objects_numerically(tile_names))
    positions = ", ".join(sort_objects_numerically(position_names))

    return f"""
You are a visual reasoning agent for a robotic planning system.

KNOWN OBJECTS (complete lists, do NOT add or remove):
- Tiles (numbered squares): {tiles} (type=tile)
- Positions (grid cells): {positions} (type=position)

IMPORTANT CONSTRAINTS ON OBJECTS:
- Tiles names encode their numbers: t_K is the tile showing number K.
- Positions are named p_I_J where I = column (left→right), J = row (top→bottom), i.e., p_1_1 is the top-left cell,
    p_2_1 is the cell to its right, p_1_2 is the cell below it, etc.

Your task is to extract **all grounded binary predicates** from the image and assign a
**confidence score** to each. Use ONLY the objects and predicate forms defined below.

Each argument must include the object name and its type, separated by a colon
(e.g. t_1:tile, p_1_2:position). Do NOT invent new predicates, objects, or types.

Valid predicate forms:

- empty(y:position)            → y is the blank (white) position with no numbered tile
- at(x:tile,y:position)        → tile x is located at position y
- neighbor(p1:position,p2:position) → p1 and p2 share an edge (no diagonals; symmetric)

For each predicate you output, assign a confidence score expressing how certain you are
that the predicate holds in the image:

- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → UNCERTAIN whether the predicate holds; visual evidence is ambiguous or unclear.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.


☑️ You MUST assign a score to **every valid predicate**, including all `at(...)` and `neighbor(...)` predicates.
Notice that you don't have to compute neighbor(x,y) for x=y, only for x≠y.

RULES:
- neighbor(...) is purely structural:
    * True (score 2) if positions differ by exactly one step horizontally or vertically.
    * False (score 0) otherwise.
    * Output both directions for true neighbors.
- Positions follow p_I_J: I = column (left→right), J = row (top→bottom).

OUTPUT FORMAT (strict):
- One predicate per line.
- Format:
    <predicate>: <score>

Valid examples:
    at(t_3:tile,p_2_1:position): 2
    empty(p_3_3:position): 2
    neighbor(p_1_2:position,p_1_3:position): 2
    neighbor(p_1_1:position,p_3_3:position): 0

Output ONLY predicate lines. No explanations.
"""