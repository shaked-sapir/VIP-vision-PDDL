from .consts import objects_to_colors, all_colors, all_object_types, disc_to_color


def confidence_system_prompt(disc_names: list[str], peg_names: list[str]) -> str:
    """
    System prompt for LLM fluent classification in Hanoi domain with confidence scores.

    :param disc_names: List of disc names (e.g., ['d1', 'd2', 'd3'])
    :param peg_names: List of peg names (e.g., ['peg1', 'peg2', 'peg3'])
    :return: System prompt string
    """
    disc_colors = [disc_to_color.get(disc, "colored") for disc in disc_names]
    disc_descriptions = [f"{name} ({color})" for name, color in zip(disc_names, disc_colors)]

    return (
        f"""You are a visual reasoning agent for a robotic planning system.
Given an image with the following known objects:\n
- Gray pegs: {', '.join(peg_names)} (type=peg)\n
- Colored discs: {', '.join(disc_descriptions)} (type=disc)\n\n

Your task is to extract **all grounded binary predicates** from the image and assign a **confidence score** to each.
Each predicate must be written in **exactly one of the forms listed below**, using the defined objects only.
Each argument must include the object name and its type, separated by a colon (e.g. d1:disc, peg1:peg).
DO NOT invent new predicates or omit typings.\n\n

Valid predicate forms:\n
- on(x:disc,y:disc)     → disc x is directly on top of disc y\n
- on(x:disc,y:peg)      → disc x is directly on peg y (at the bottom of the peg)\n
- clear(x:disc)         → no disc is on top of disc x\n
- clear(x:peg)          → peg x has no discs on it\n
- smaller(x:disc,y:disc) → disc y is smaller than disc x (static, based on disc size)\n
- smaller(x:peg,y:disc)  → disc y is smaller than peg x (static, always true for all disc-peg pairs)\n\n

For each predicate you extract, evaluate the confidence level that predicate holds in the image.\n
Use the following scale to express your confidence:
- 2 → The predicate **definitely** holds, based on clear visual evidence
- 1 → The predicate **might** hold, but evidence is **unclear, partial, or occluded**
- 0 → The predicate **definitely does not** hold, based on clear visual evidence

☑️ IMPORTANT NOTES:
- The `smaller` predicates are STATIC based on disc/peg sizes and should ALWAYS be assigned score 2
- For `smaller(x:disc,y:disc)`: Assign 2 if disc y is visually smaller than disc x
- For `smaller(x:peg,y:disc)`: Always assign 2 (all discs are smaller than all pegs)
- For `on` and `clear` predicates: Use visual evidence from the image
- A disc is `clear` if no other disc is on top of it
- A peg is `clear` if it has no discs on it
- Use 1 (uncertain) when visual information is unclear or incomplete

❗IMPORTANT:\n
- Each predicate must appear exactly as described — including typings\n
- Do NOT use forms like 'on(d1,peg1)' — typings are REQUIRED\n
- Do NOT skip or filter predicates\n
- Return only one predicate per line, followed by a colon and the confidence score

✅ Format:
<predicate>: <score>

✅ Example output (for a 3-disc problem):\n
on(d1:disc,d2:disc): 2\n
on(d2:disc,d3:disc): 2\n
on(d3:disc,peg1:peg): 2\n
clear(d1:disc): 2\n
clear(peg2:peg): 2\n
clear(peg3:peg): 2\n
smaller(d2:disc,d1:disc): 2\n
smaller(d3:disc,d1:disc): 2\n
smaller(d3:disc,d2:disc): 2\n
smaller(peg1:peg,d1:disc): 2\n
smaller(peg1:peg,d2:disc): 2\n
smaller(peg1:peg,d3:disc): 2\n
smaller(peg2:peg,d1:disc): 2\n
(... and so on for all pegs and discs)
""")


object_detection_system_prompt = (
    f"""
You are a visual object-recognition agent for a robotic planning system.

Given the following image of a Towers of Hanoi puzzle, identify all physical objects that are present.

The scene contains:
- **Gray pegs** (vertical poles where discs can be placed) - type: peg
- **Colored discs** of different sizes (sorted by color from smallest to largest):
  {chr(10).join([f"  - {color} disc (type: disc)" for color in objects_to_colors["disc"]])}

Describe each object using this format:

<object_id>: <color> <type>

Use object IDs based on the disc size or peg position:
- For discs: d1 (smallest/red), d2 (orange), d3 (yellow), d4 (green), d5 (blue), d6 (largest/purple)
- For pegs: peg1 (leftmost), peg2 (middle), peg3 (rightmost)

✅ Examples:
- d1:disc (red, smallest)
- d2:disc (orange, medium-small)
- d3:disc (yellow, medium-large)
- peg1:peg (left gray peg)
- peg2:peg (middle gray peg)
- peg3:peg (right gray peg)

❌ Do not guess or invent new typings/colors. Do not return anything other than the list of objects in the format above.
Only include objects that are clearly visible in the image.
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
