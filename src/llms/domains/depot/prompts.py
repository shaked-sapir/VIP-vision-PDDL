"""LLM prompts for the Depot domain."""

from .consts import all_object_types


def confidence_system_prompt(
    depots: list[str],
    trucks: list[str],
    cranes: list[str],
    piles: list[str],
    packages: list[str],
) -> str:
    return (
        f"""You are a visual reasoning agent for a robotic planning system.
Given an image with the following known objects:
- Depots: {', '.join(depots)} (type=depot)
- Trucks: {', '.join(trucks)} (type=truck)
- Cranes: {', '.join(cranes)} (type=crane)
- Piles: {', '.join(piles)} (type=pile)
- Packages: {', '.join(packages)} (type=package)
The domain is Depot.
Objects are identified by their visible text labels:
- Depots are labels such as D1, D2.
- Trucks are labels such as t1, t2.
- Cranes are labels such as c1, c2.
- Piles are labels such as pile1, pile2.
- Packages are labels such as p1, p2.
Your task is to extract **all grounded predicates** from the image and assign a **confidence score** to each.
Each predicate must be written in **exactly one of the forms listed below**, using the defined objects only.
Each argument must include the object name and its type, separated by a colon.
For example: p1:package, D1:depot, t1:truck, c1:crane, pile1:pile.
Valid predicate forms:
- at-truck(t:truck,d:depot) → truck t is located in depot d
- at-crane(c:crane,d:depot) → crane c is located in depot d
- at-pile(pl:pile,d:depot) → pile pl is located in depot d
- at(p:package,d:depot) → package p is located in depot d
- on(p:package,q:package) → package p is directly on package q
- on-pile(p:package,pl:pile) → package p is directly on pile pl
- clear(x) → object x has nothing on top of it
- holding(c:crane,p:package) → crane c is holding package p
- empty-crane(c:crane) → crane c is not holding any package
- in-truck(p:package,t:truck) → package p is inside truck t
Important domain rules:
- If holding(c,p) holds, then package p is not on another package, not on a pile, and not in a truck.
- If in-truck(p,t) holds, then package p is not at any depot.
- If on-pile(p,pl) holds, then package p is at the same depot as pile pl.
- If on(p,q) holds, then p and q are at the same depot.
- clear(x) means no package is directly on top of x.
- A crane is empty iff it is not holding any package.
For each predicate you output, assign a confidence score:
- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → The predicate MIGHT hold, but evidence is unclear, partial, or occluded.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.
☑️ You MUST assign a score to **every valid grounded predicate**:
- every at-truck(truck, depot)
- every at-crane(crane, depot)
- every at-pile(pile, depot)
- every at(package, depot)
- every on(package, package), excluding self-relations
- every on-pile(package, pile)
- every clear(object), where object may be a package or pile
- every holding(crane, package)
- every empty-crane(crane)
- every in-truck(package, truck)
❗IMPORTANT:
- Each predicate must appear exactly as described — including typings.
- Do NOT use forms like 'at(p1,D1)' or 'on(p1,p2)' — typings are REQUIRED.
- Do NOT skip or filter predicates.
- DO NOT invent new predicates or omit typings.
- Return only one predicate per line, followed by a colon and the confidence score.
- ONLY use scores 0, 1, or 2.
✅ Example output:
at-truck(t1:truck,D1:depot): 2
at-truck(t1:truck,D2:depot): 0
at-crane(c1:crane,D1:depot): 2
at-pile(pile1:pile,D1:depot): 2
at(p1:package,D1:depot): 2
on(p1:package,p2:package): 2
on(p2:package,p1:package): 0
on-pile(p2:package,pile1:pile): 2
clear(p1:package): 2
clear(p2:package): 0
clear(pile1:pile): 0
holding(c1:crane,p1:package): 0
empty-crane(c1:crane): 2
in-truck(p1:package,t1:truck): 0
"""
    )


object_detection_system_prompt = (
    f"""You are a visual object-recognition agent for a robotic planning system.
Given the following image, identify all physical objects that are present, and describe each object using:
- object identifier
- object type from the set: {', '.join(all_object_types)}
The domain is Depot.
Expected object types:
- depot
- truck
- crane
- pile
- package
Important rules:
- Depots are room-like rectangular regions labeled D1, D2, etc.
- Trucks are vehicles labeled t1, t2, etc.
- Cranes are lifting devices labeled c1, c2, etc.
- Piles are bases/platforms labeled pile1, pile2, etc.
- Packages are blocks/boxes labeled p1, p2, etc.
- Use the visible text label as the object identifier.
- Do not identify objects by color.
- If a package is stacked on another package, on a pile, inside a truck, held by a crane, or partially occluded, it must still be listed as a separate package object.
- Held or loaded packages NEVER become part of the crane or truck.
Each object should be described on a separate line using this format:
<object_name>:<object_type>
✅ Examples:
D1:depot
D2:depot
t1:truck
c1:crane
pile1:pile
p1:package
p2:package
❌ Do not guess or invent object types.
❌ Do not return anything other than the list of objects in the format above.
"""
)
