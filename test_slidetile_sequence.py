"""Random 10-step Slidetile (PDDLGym) trace → images + .trajectory file."""

from copy import deepcopy
from pathlib import Path

import pddlgym
from PIL import Image
from pddlgym.core import _select_operator

PROBLEM_INDEX = 1
NUM_STEPS = 10
OUTPUT_DIR = Path(__file__).parent / "slidetile_test_sequence"


def to_pddl_lit(literal: str) -> str:
    name, args = literal.split("(", 1)
    args = args.rstrip(")")
    if not args:
        return f"({name.strip()})"
    parts = [a.strip().split(":")[0] for a in args.split(",")]
    return f"({name.strip()} {' '.join(parts)})"


def to_pddl_action(action: str) -> str:
    name, args = action.split("(", 1)
    args = args.rstrip(")")
    parts = [a.strip().split(":")[0] for a in args.split(",")] if args else []
    return f"({name.strip()} {' '.join(parts)})".replace("  ", " ")


def ground_action(operator, assignment) -> str:
    return f"{operator.name}({', '.join(str(assignment[p]) for p in operator.params)})"


def sample_state_changing_action(env, state, allow_done: bool = False):
    """Sample a random action that changes the state (same logic as ImageTrajectoryHandler)."""
    while True:
        action = env.action_space.sample(state)
        tmp = deepcopy(env)
        new_state, _, done, _, _ = tmp.step(action)
        if new_state == state or (done and not allow_done):
            continue
        _ = env.action_space.sample(new_state)
        new_state, _, done, _, _ = env.step(action)
        return action, new_state, done


def build_trajectory_file(steps) -> str:
    """Format: (:init ...) (operator: ...) (:state ...) ... matching project .trajectory files."""
    lines = [
        "(",
        f"(:init {' '.join(to_pddl_lit(str(l)) for l in steps[0]['current'].literals)})",
    ]
    for i, step in enumerate(steps):
        if i > 0:
            lines.append(f"(:state {' '.join(to_pddl_lit(str(l)) for l in step['current'].literals)})")
        lines.append(f"(operator: {to_pddl_action(step['ground_action'])})")
    lines.append(f"(:state {' '.join(to_pddl_lit(str(l)) for l in steps[-1]['next'].literals)})")
    lines.append(")")
    return "\n".join(lines)


def save_state_image(env, output_dir: Path, index: int) -> None:
    img = env.render(mode="rgb_array")
    if img.dtype != "uint8":
        img = (img * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(output_dir / f"state_{index:04d}.png")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    trajectory_path = OUTPUT_DIR / f"problem{PROBLEM_INDEX}.trajectory"

    env = pddlgym.make("PDDLEnvSlidetile-v0")
    env.fix_problem_index(PROBLEM_INDEX)
    state, _ = env.reset()
    problem_path = env.problems[PROBLEM_INDEX].problem_fname

    print(f"PDDLEnvSlidetile-v0  problem_index={PROBLEM_INDEX}  ({Path(problem_path).name})")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Initial state: {sorted(to_pddl_lit(str(l)) for l in state.literals)}\n")

    save_state_image(env, OUTPUT_DIR, 0)
    print("Saved state_0000.png\n")

    steps = []
    for step_idx in range(1, NUM_STEPS + 1):
        prev = state
        allow_done = step_idx == NUM_STEPS
        action, state, done = sample_state_changing_action(env, prev, allow_done=allow_done)
        op, assignment = _select_operator(prev, action, env.domain)
        ga = ground_action(op, assignment)
        pddl_action = to_pddl_action(ga)

        save_state_image(env, OUTPUT_DIR, step_idx)

        print(f"Step {step_idx}: {pddl_action}")
        print(f"  next state: {sorted(to_pddl_lit(str(l)) for l in state.literals)}")
        print(f"  saved state_{step_idx:04d}.png\n")

        steps.append({"current": prev, "ground_action": ga, "next": state})
        if done:
            print("Goal reached on final step.")

    trajectory = build_trajectory_file(steps)
    trajectory_path.write_text(trajectory)
    print(f"Wrote {trajectory_path}\n")
    print(trajectory)


if __name__ == "__main__":
    main()
