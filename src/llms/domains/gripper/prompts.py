"""LLM prompts for the Gripper domain."""

from .consts import all_object_types


def confidence_system_prompt(
    ball_ids: list[str],
    rooms: list[str],
    grippers: list[str],
) -> str:
    """Returns the fluent classification prompt for the Gripper domain.

    Args:
        ball_ids: List of ball names (e.g. ["ball1", "ball2"]).
        rooms: List of room names (e.g. ["rooma", "roomb"]).
        grippers: List of gripper names (e.g. ["left", "right"]).
    """
    return (
        f"""You are a visual reasoning agent for a robotic planning system.
Given an image with the following known objects:
- Rooms: {', '.join(rooms)} (type=room)
- Grippers: {', '.join(grippers)} (type=gripper)
- Balls: {', '.join(ball_ids)} (type=ball)
The domain is Gripper.
The robot is named robby.
The robot has two grippers:
- R means right gripper
- L means left gripper
Balls are identified by the numeric identifier printed on them, e.g. ball1, ball2.
All balls may have the same color, so do NOT use color as identity.
Your task is to extract **all grounded predicates** from the image and assign a **confidence score** to each.
Each predicate must be written in **exactly one of the forms listed below**, using the defined objects only.
Each argument must include the object name and its type, separated by a colon, e.g. ball1:ball, rooma:room, left:gripper.
DO NOT invent new predicates or omit typings.
Valid predicate forms:
- at-robby(r:room) → robby is located in room r
- at(b:ball,r:room) → ball b is located in room r
- free(g:gripper) → gripper g is empty
- carry(b:ball,g:gripper) → gripper g is carrying ball b
Important domain rule:
- If carry(b:ball,g:gripper) holds, then at(b:ball,r:room) does NOT hold for any room r.
- A ball is either located in a room or carried by a gripper, not both.
- A gripper is free iff it is not carrying any ball.
For each predicate you output, assign a confidence score expressing how certain you are that the predicate holds in the image:
- 2 → The predicate DEFINITELY holds, based on clear visual evidence.
- 1 → The predicate MIGHT hold, but evidence is unclear, partial, or occluded.
- 0 → The predicate DEFINITELY does NOT hold, based on clear visual evidence.
☑️ You MUST assign a score to **every valid grounded predicate**:
- every at-robby(room)
- every at(ball, room)
- every free(gripper)
- every carry(ball, gripper)
❗IMPORTANT:
- Each predicate must appear exactly as described — including typings.
- Do NOT use forms like 'at(ball1,rooma)' or 'carry(1,left)' — typings are REQUIRED.
- Do NOT skip or filter predicates.
- DO NOT invent new predicates or omit typings. stick to the rules above.
- Return only one predicate per line, followed by a colon and the confidence score.
- ONLY use scores 0, 1, or 2.
✅ Example output:
at-robby(rooma:room): 2
at-robby(roomb:room): 0
at(ball1:ball,rooma:room): 2
at(ball1:ball,roomb:room): 0
carry(ball1:ball,left:gripper): 0
carry(ball1:ball,right:gripper): 0
free(left:gripper): 2
free(right:gripper): 2
"""
    )


object_detection_system_prompt = (
    f"""You are a visual object-recognition agent for a robotic planning system.
Given the following image, identify all physical objects that are present, and describe each object using:
- object identifier
- object type from the set: {', '.join(all_object_types)}
The domain is Gripper.
Expected object types:
- room
- ball
- gripper
Important rules:
- The robot is named robby, but it is NOT a typed object in the PDDL domain.
- Rooms are named by their visible labels, for example rooma and roomb.
- The robot has two grippers:
  - R means right gripper
  - L means left gripper
- Balls are identified by the numeric identifier printed on them.
  For example, a ball with the number 1 should be written as ball1:ball.
- Do not use ball color as identity, because all balls may have the same color.
- If a ball is held by a gripper, partially hidden, touching the gripper, or occluded by the robot, it must still be listed as a separate ball object.
- Held objects NEVER become part of the robot or gripper.
Each object should be described on a separate line using this format:
<object_name>:<object_type>
✅ Examples:
rooma:room
roomb:room
left:gripper
right:gripper
ball1:ball
ball2:ball
❌ Do not guess or invent object types.
❌ Do not return anything other than the list of objects in the format above.
"""
)
