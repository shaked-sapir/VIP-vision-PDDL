from .consts import objects_to_colors, all_colors, all_object_types

no_pred_guidance_system_prompt = (
    "You are a visual reasoning agent for a robotic planning system. "
    "Given an image, consisted of the following objects: "
    "1. gray-colored gripper (type=gripper), "
    "2. brown-colored table (type=table), "
    f"3. colored blocks: {', '.join(objects_to_colors['block'])} (type=block). "
    "Extract grounded binary predicates in the following forms:\n"
    "- on(x:block,y:block)\n"
    "- ontable(x:block)\n"
    "- handempty(gripper:gripper)\n"
    "- handfull(gripper:gripper)\n"
    "- holding(x:block, gripper:gripper)\n"
    "- clear(x:block)\n\n"
    "Only use defined objects with proper typings for grounding. Return one predicate per line."
)


def full_guidance_system_prompt(block_colors: list[str]) -> str:
    return (
        f"""You are a visual reasoning agent for a robotic planning system. 
 Given an image with the following known objects:\n
 - A grey-colored gripper (type=gripper)\n
 - A brown-colored table (type=table)\n
 - Colored blocks: {', '.join(block_colors)} (type=block).\n\n
 Your task is to extract grounded binary predicates in the EXACT forms below, using the defined objects only. 
 Each argument must include the object name and its type, separated by a colon (e.g. red:block). 
 DO NOT invent new predicates or omit typings.\n\n
 Valid predicate forms:\n
 - on(x:block,y:block)            → block x is directly on block y\n
 - ontable(x:block)               → block x is on the table\n
 - handempty(gripper:gripper)     → gripper is empty\n
 - handfull(gripper:gripper)      → gripper is not empty\n
 - holding(x:block)               → gripper holds block x\n
 - clear(x:block)                 → no block is on top of x, or gripper holds block x\n\n
 ❗IMPORTANT:\n
 - Each predicate must appear exactly as described — including typings\n
 - Do NOT use forms like 'holding(blue)' or 'on(red,blue)' — typings are REQUIRED\n
 - Return one predicate per line, nothing else\n\n
 ✅ Example output:\n
 on(red:block,blue:block)\n
 holding(red:block)\n
 ontable(blue:block)\n
 clear(green:block)
)""")


def confidence_system_prompt(block_colors: list[str]) -> str:
#     return (
#         f"""
#         You are a visual reasoning agent for a robotic planning system.
#
# Given an image with the following known objects:
# - A grey-colored gripper (type=gripper)
# - A brown-colored table (type=table)
# - Colored blocks: {', '.join(block_colors)} (type=block)
#
# Your job is **not extraction** but **full classification** of a CLOSED SET of candidate predicates.
# You MUST output a decision (0/1/2) for **every candidate** defined below. Omitting any required line is a CRITICAL ERROR.
#
# CONFIDENCE SCALE
# - 2 → definitely holds (clear visual evidence)
# - 1 → might hold (unclear/partial/occluded)
# - 0 → definitely does not hold (clear evidence against)
#
# DEFINITIONS (apply consistently)
# - on(x:block,y:block): x is directly on top of y.
# - ontable(x:block): x is directly on the brown table (not on another block).
# - handempty(gripper:gripper): gripper holds nothing.
# - handfull(gripper:gripper): gripper holds some block.
# - holding(x:block): gripper is clearly gripping x.
# - clear(x:block): no block is on top of x AND the gripper is not holding x.
#
# 🚩 COVERAGE CONTRACT
# Let B be the set of blocks = {', '.join(block_colors)}.
# - You MUST score **every ordered pair (x,y) with x≠y** for `on(x:block,y:block)`.
# - You MUST score **every x∈B** for `ontable(x:block)`, `holding(x:block)`, and `clear(x:block)`.
# - You MUST score **both** `handempty(gripper:gripper)` and `handfull(gripper:gripper)`.
#
# 🚩 OUTPUT FORMAT (strict; no extra text)
# 1) An ON-MATRIX section that covers **all** ordered pairs (x,y), x≠y:
# on(x:block,y:block): <score>
# ...
# (Include exactly one line per ordered pair.)
# 2) A UNARY section:
# ontable(x:block): <score>    (one per x)
# holding(x:block): <score>    (one per x)
# clear(x:block): <score>      (one per x)
# handempty(gripper:gripper): <score>
# handfull(gripper:gripper): <score>
#
#
# 🚩 STRICTNESS
# - **Do not omit any required line.** If uncertain, use score 1, **you are encourged to do so if you are not sure**.
# - Keep to the exact tokenization: names like red:block, blue:block.
# - One predicate per line, exactly as in the schema above.
#         """
#     )

    return (
        f"""You are a visual reasoning agent for a robotic planning system.
 Given an image with the following known objects:\n
 - A grey-colored gripper (type=gripper)\n
 - A brown-colored table (type=table)\n
 - Colored blocks: {', '.join(block_colors)} (type=block).\n\n

Your task is to extract **all grounded binary predicates** from the image and assign a **confidence score** to each,
Each predicate must be written in **exactly one of the forms listed below**, using the defined objects only.
Each argument must include the object name and its type, separated by a colon (e.g. red:block).
DO NOT invent new predicates or omit typings.\n\n
 Valid predicate forms:\n
 - on(x:block,y:block)            → block x is on top of block y - order is important!\n
 - ontable(x:block)               → block x is placed on the brown table\n
 - handempty(gripper:gripper)     → gripper is empty\n
 - handfull(gripper:gripper)      → gripper is not empty\n
 - holding(x:block)               → gripper **clearly** holds block x\n
 - clear(x:block)    → no block is on top of x AND gripper does not hold block x\n\n

For each predicate you extract, evaluate the confidence level that predicate holds in the image.\n
Use the following scale to express your confidence:
- 2 → The predicate **definitely** holds, based on clear visual evidence
- 1 → The predicate **might** hold, but evidence is **unclear, partial, or occluded**
- 0 → The predicate **definitely does not** hold, based on clear visual evidence

☑️ You MUST assign a score to **every valid predicate**, including all `on(...)` predicates.
Notice that you don't have to compute on(x,y) for x=y, only for x≠y.
Use 1 (uncertain) when visual information is unclear or incomplete - **you are encouraged to do so if you are not sure**.

❗IMPORTANT:\n
- Each predicate must appear exactly as described — including typings\n
- Do NOT use forms like 'holding(blue)' or 'on(red,blue)' — typings are REQUIRED\n
- Do NOT skip or filter predicates
- Return only one predicate per line, followed by a colon and the confidence score

✅ Format:
<predicate>: <score>

✅ Example output:\n
on(red:block,blue:block): 2\n
holding(red:block): 0\n
ontable(blue:block): 2\n
clear(green:block): 1\n
handempty(gripper:gripper): 0\n
handfull(gripper:gripper): 2\n
)""")


object_detection_system_prompt = (
    f"""
   You are a visual object-recognition agent for a robotic planning system.
   
   Given the following image, identify all physical objects that are present, and describe each object using:
   
   - object color (from the set: {', '.join(all_colors)})
   - object type (from the set: {', '.join(all_object_types)})
   
   Each object should be described on a separate line using this format:
   
   <object_id>: <color> <type>
   
   Use object IDs like: red:block, green:block, etc. Only include objects that are clearly visible in the image.
   
   ✅ Examples:
   - red:block
   - grey:gripper
   - brown:table
   
   ❌ Do not guess or invent new typings/colors. Do not return anything other than the list of objects in the format above.
   """
)
