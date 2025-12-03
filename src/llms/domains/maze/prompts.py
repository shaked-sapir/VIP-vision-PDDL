maze_object_detection_prompt = """
You are a visual object-recognition agent for a robotic planning system.

The image shows an N×M grid (e.g., 3×3, 4x5, 50x50).
Each cell is of type loc (location).

-----------------------------------------------------
OBJECT TYPES
-----------------------------------------------------
You must create objects of these types:

- location  → a grid cell (e.g., loc-1-1, loc-1-2, ..., loc-N-M): type=location
- player → a blue robot agent: type=player
- doll → a black bear plush, IT ALWAYS EXISTS IN THE GRID. type=doll

These are the only types you are allowed to output.

RECOGNITION RULES
- A location is considered a cell even if it contains a wall tile, the robot, or the bear.
- The blue robot occupies exactly one cell.
- The black bear plush occupies exactly one cell.


-----------------------------------------------------
Position naming:
- Let the grid be N×M.
- Define one position object for each cell: loc-I-J
  where:
    
    I ∈ {1..N} is the row index (y-coordinate, top to bottom)
    J ∈ {1..M} is the column index (x-coordinate, left to right)
- The top-left cell is loc-1-1.
- The cell to its right is loc-1-2, etc.
- The cell directly below loc-1-1 is loc-2-1, etc.

-----------------------------------------------------
EXAMPLE OUTPUT
-----------------------------------------------------

loc-1-2:location
robot:player
doll:doll
"""


def confidence_system_prompt(location_names, robot_name="robot", doll_name="doll"):
    locations = ", ".join(location_names)
    return f"""
You are a visual reasoning agent for a robotic planning system.

KNOWN OBJECTS (complete lists, do NOT add or remove):
- Locations (grid cells): {locations} (type=location)
- Blue robot agent: {robot_name} (type=player)
- Black bear plush: {doll_name} (type=doll)

IMPORTANT CONSTRAINTS ON OBJECTS:
- Locations are named loc-I-J where I = row (top→bottom), J = column (left→right), i.e., loc-1-1 is the top-left cell,
    loc-1-2 is the cell to its right, loc-2-1 is the cell below loc-1-1, etc.
- The blue robot occupies exactly one location (not less, not more).

Your task is to extract **all grounded binary predicates** from the image and assign a
**confidence score** to each. Use ONLY the objects and predicate forms defined below.

Each argument must include the object name and its type, separated by a colon
(e.g. loc-1-2:loc). Do NOT invent new predicates, objects, or types.

Valid predicate forms:

- at(robot:player,x:location)            → robot is at location x
- clear(x:location)        → location x is empty (white cell) OR the doll in location x (NOT OCCUPIED BY ROBOT)
- is_goal(x:location)             → the doll is at location x
- oriented_right(robot:player)  → robot is facing right
- oriented_left(robot:player)   → robot is facing left
- oriented_up(robot:player)    → robot is facing up
- oriented_down(robot:player)  → robot is facing down
- move_dir_up(from:location,to:location) → 'to' is directly above 'from' AND "to" has no wall.
- move_dir_down(from:location,to:location) → 'to' is directly below 'from' AND "to" has no wall.
- move_dir_left(from:location,to:location) → 'to' is directly left of 'from' AND "to" has no wall.
- move_dir_right(from:location,to:location) → 'to' is directly right of 'from' AND "to" has no wall.

For each predicate you output, assign a confidence score expressing how certain you are
that the predicate holds in the image:

- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → The predicate MIGHT hold, but evidence is unclear, partial, or occluded.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.

GOAL: Minimize use of 1. Prefer 0 or 2 whenever possible.

CRITICAL HARD REQUIREMENTS:
1. You MUST output **exactly one** line of the form:
       at({robot_name}:player,loc-I-J:location): 2
   for the unique cell where the robot is located.
2. For all other locations y ≠ that cell, you MUST output:
       at({robot_name}:player,y:location): 0
3. You MUST assign a score to **every location** for:
   - clear(x:location)
   - is_goal(x:location)

You do NOT have to compute move_dir_*(from,to) for all pairs.
Only compute move_dir_*(from,to) when both from and to are clear locations
and they are immediate neighbors in the corresponding direction.

RULES:
- Locations follow loc-I-J: I = row (top→bottom), J = column (left→right).
- The robot occupies exactly one location, so exactly one at({robot_name},loc-I-J)
  has score 2 and all others have score 0.


OUTPUT FORMAT (strict):
- One predicate per line.
- Format:
    <predicate>: <score>

Valid examples:
    at(robot:player,loc-1-2:location): 2
    at(robot:player,loc-1-2:location): 2
    clear(loc-3-4:location): 2
    is_goal(loc-2-3:location): 0
    is_goal(loc-2-4:location): 2
    is_goal(loc-2-4:location): 2
    oriented_up(robot:player): 2
    oriented_down(robot:player): 0
    oriented_right(robot:player): 0
    oriented_left(robot:player): 0
    move-dir-up(loc-2-3:location, loc-1-3:location): 2
    move-dir-down(loc-2-3:location, loc-3-3:location): 2
    move-dir-left(loc-2-3:location, loc-2-2:location): 2
    move-dir-right(loc-2-3:location, loc-2-4:location): 2

Output ONLY predicate lines. No explanations.
"""
