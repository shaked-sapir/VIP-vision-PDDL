maze_object_detection_prompt = """
You are a visual object-recognition agent for a robotic planning system.

The image shows an N×M grid (e.g., 3×3, 4x5, 50x50).
Each cell is of type loc (location).

-----------------------------------------------------
OBJECT TYPES
-----------------------------------------------------
You must create objects of these types:

- location  → a grid cell (e.g., loc-1-1, loc-1-2, ..., loc-N-M): type=location
- robot → a cartoon turquoise/blue robot with a square head, orange chest panel,
  and two small arms. It may be rotated but is always the only blue robot in the grid.
- doll → a black bear plush, IT ALWAYS EXISTS IN THE GRID. type=doll

These are the only types you are allowed to output.

RECOGNITION RULES
- A location is considered a cell even if it contains a wall tile, the robot, or the bear.
- The robot is BLUE and occupies exactly one cell in the grid.
- The black bear plush occupies exactly one cell.


-----------------------------------------------------
Position naming:
- Let the grid be N×M.
- Define one position object for each cell: loc_I_J
  where:
    
    I ∈ {1..N} is the row index (y-coordinate, top to bottom)
    J ∈ {1..M} is the column index (x-coordinate, left to right)
- The top-left cell is loc_1_1.
- The cell to its right is loc_1_2, etc.
- The cell directly below loc_1_1 is loc_2_1, etc.

-----------------------------------------------------
EXAMPLE OUTPUT
-----------------------------------------------------

loc_1_2:location
robot:robot
doll:doll
"""


def confidence_system_prompt(location_names, robot_name="robot", doll_name="doll"):
    locations = ", ".join(location_names)
    return f"""
You are a visual reasoning agent for a robotic planning system.

KNOWN OBJECTS (complete lists, do NOT add or remove):
- Locations (grid cells): {locations} (type=location)
- ROBOT: A cartoon turquoise/blue robot with a square head, orange chest panel,
  and two small arms: {robot_name} (type=robot)
- Black bear plush: {doll_name} (type=doll)

-----------------------------------------------------
VISUAL DEFINITIONS (IMPORTANT)
-----------------------------------------------------
- WALL cell: brown/olive tile with a brick pattern.
- FLOOR cell: white/bright tile with no brick texture.
- ROBOT cell: the turquoise/blue robot sprite.
- DOLL cell: the black bear plush.

A cell is CLEAR iff it is FLOOR or contains the DOLL.
A cell is NOT clear iff it is WALL or contains the ROBOT.

=====================================================
BLOCK 1 — PERCEPTION (MANDATORY)
=====================================================
For EVERY location loc_I_J output EXACTLY one line:

    cell(loc_I_J:location) = {{floor | wall | robot | doll}}

1. For every cell that is NOT a wall and NOT empty, describe the object briefly:
   - (row, col): short description of colors and shape.
2. From those descriptions, decide which cell contains the ROBOT
   (blue/turquoise rectangular body, cyan face, red antennas).
3. Hold the location of the ROBOT in memory to output later, in this json format:
 {{"robot_at": [row, col]}}
 
 Notice that that ROBOT cannot be in a WALL cell. 
 if you decide its on wall cell - then it means you made a mistake in perception, and you should correct it.



=====================================================
BLOCK 2 — PREDICATES WITH SCORES
=====================================================

(1) Robot and goal:
- at({robot_name}:robot, x:location): 2   iff   the center of the turquoise/blue robot with a square head,
    orange chest panel, and two small arms is at location x.
  (Do NOT output the 0-score at() lines; omitted means score 0.)
- is_goal(x:location): 2   iff   doll ∈ cell(x)

(2) Clear:
- clear(x:location): 2  iff cell(x)=floor [completely white] or cell(x)=doll 
- clear(x:location): 0  iff cell(x)=wall or cell(x)=robot

(3) Orientation (choose ONE with score 2, others 0 or omitted)
- oriented_up({robot_name}:robot): 2  iff  the robot is facing UP
- oriented_down({robot_name}:robot): 2  iff  the robot is facing DOWN
- oriented_left({robot_name}:robot): 2  iff  the robot is facing LEFT
- oriented_right({robot_name}:robot): 2  iff  the robot is facing RIGHT

(4) Directional movement:
Let FROM = loc_I1_J1, TO = loc_I2_J2.

move_dir_up(FROM,TO): 2  iff
    I2 = I1 - 1  AND J2 = J1  AND  FLOOR(FROM)=2  AND FLOOR(TO)=2 

move_dir_down(FROM,TO): 2 iff
    I2 = I1 + 1  AND J2 = J1  AND  FLOOR(FROM)=2  AND FLOOR(TO)=2

move_dir_left(FROM,TO): 2 iff
    I2 = I1  AND J2 = J1 - 1  AND  FLOOR(FROM)=2  AND  FLOOR(TO)=2

move_dir_right(FROM,TO): 2 iff
    I2 = I1  AND J2 = J1 + 1  AND  FLOOR(FROM)=2  AND FLOOR(TO)=2

Output these ONLY for immediate neighbors and ONLY when both FROM and TO are floor tiles (white and not wall)
even if they are not clear (if robot is at one of them-you should still output).
If a predicate does not hold, you may omit it (it is treated as score 0).


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
