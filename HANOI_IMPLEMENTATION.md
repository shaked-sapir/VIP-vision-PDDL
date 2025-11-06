# Hanoi Domain Implementation

This document describes the complete implementation of object detection and fluent classification for the Hanoi domain, including both deterministic and LLM-based approaches.

## Overview

The Hanoi domain implementation follows the same architecture as the Blocks domain, with both **deterministic** (rule-based) and **LLM-based** (GPT-4 Vision) variants for object detection and fluent classification.

## File Structure

```
src/
├── llms/
│   └── domains/
│       └── hanoi/
│           ├── __init__.py              # Package init
│           ├── consts.py                # Color mappings and constants
│           └── prompts.py               # LLM system prompts
│
├── object_detection/
│   ├── hanoi_object_detector.py         # Deterministic detector (position/size-based)
│   └── llm_hanoi_object_detector.py     # LLM-based detector (GPT-4 Vision)
│
├── fluent_classification/
│   ├── hanoi_fluent_classifier.py       # Deterministic classifier (geometric)
│   └── llm_hanoi_fluent_classifier.py   # LLM-based classifier (GPT-4 Vision)
│
└── trajectory_handlers/
    ├── hanoi_image_trajectory_handler.py     # Deterministic trajectory handler
    └── llm_hanoi_trajectory_handler.py       # LLM-based trajectory handler
```

## Domain Characteristics

### Objects

The Hanoi domain consists of:
- **Pegs**: Gray vertical poles (typically 3: peg1, peg2, peg3)
- **Discs**: Colored horizontal discs of varying sizes (d1 to d6, where d1 is smallest)

### Predicates

1. **on(x:disc, y:disc)** - disc x is directly on top of disc y
2. **on(x:disc, y:peg)** - disc x is on peg y (at the bottom)
3. **clear(x:disc)** - no disc is on top of disc x
4. **clear(x:peg)** - peg x has no discs on it
5. **smaller(x:disc, y:disc)** - disc y is smaller than disc x (static based on size)
6. **smaller(x:peg, y:disc)** - always true (all discs are smaller than pegs)

### Color Scheme

To distinguish discs in the default pddlgym rendering (where all discs are red), we use:
- **Position-based detection**: Identify discs by their horizontal and vertical positions
- **Size-based identification**: Sort discs by area to determine d1 (smallest) to d6 (largest)
- **Peg detection**: Identify gray vertical rectangles as pegs, sorted left-to-right

For LLM-based detection, the prompts describe colors from smallest to largest:
- d1: red (smallest)
- d2: orange
- d3: yellow
- d4: green
- d5: blue
- d6: purple (largest)

## Implementation Details

### 1. Deterministic Approach

#### Object Detection: `HanoiObjectDetector`

**Strategy**: Position and size-based contour detection

**How it works**:
1. Detect red regions (discs) using color thresholding
2. Find contours and filter by aspect ratio (width > height for discs)
3. Sort discs by area (largest to smallest)
4. Assign IDs: largest = highest number (d3), smallest = d1
5. Detect gray regions (pegs) with vertical aspect ratio (height > width)
6. Sort pegs by horizontal position (left to right)
7. Assign IDs: leftmost = peg1, middle = peg2, rightmost = peg3

**Key parameters**:
- `color_threshold`: 0.15 (tolerance for color matching)
- `min_area_threshold`: 50 pixels (minimum object size)

#### Fluent Classification: `HanoiFluentClassifier`

**Strategy**: Geometric relationship analysis

**How it works**:

**on(x, y) predicates**:
- Check horizontal alignment (x-coordinates within threshold)
- Check vertical adjacency (y2 > y1, disc below is within 100 pixels)
- Verify no disc exists between them
- For disc-on-peg: Check if disc is lowest on that peg

**clear(x) predicates**:
- For discs: Check if any disc is on top of it (using on predicates)
- For pegs: Check if any disc is on the peg

**smaller(x, y) predicates**:
- Static predicates based on disc IDs
- smaller(d3, d1) = TRUE (d1 is smaller than d3)
- smaller(peg, disc) = TRUE (always)

**Key parameters**:
- `vertical_threshold`: 5 pixels
- `horizontal_threshold`: 50 pixels

### 2. LLM-Based Approach

#### Object Detection: `LLMHanoiObjectDetector`

**Strategy**: GPT-4 Vision API with custom prompt

**System Prompt**:
- Describes the scene: gray pegs and colored discs
- Lists expected objects with naming convention (d1-d6, peg1-peg3)
- Maps disc colors to sizes (red=smallest, purple=largest)
- Requires strict format: `<object_id>: <color> <type>`

**Example Output**:
```
d1:disc (red, smallest)
d2:disc (orange)
peg1:peg (left gray peg)
```

#### Fluent Classification: `LLMHanoiFluentClassifier`

**Strategy**: GPT-4 Vision with confidence scoring

**System Prompt** (`confidence_system_prompt`):
- Describes all predicate types with exact syntax
- Specifies confidence scale (0 = false, 1 = uncertain, 2 = true)
- Explains special handling for `smaller` predicates (static, always score 2)
- Requires complete coverage of all predicate combinations

**Example Output**:
```
on(d1:disc,d2:disc): 2
on(d2:disc,peg1:peg): 2
clear(d1:disc): 2
clear(peg2:peg): 2
smaller(d2:disc,d1:disc): 2
smaller(peg1:peg,d1:disc): 2
```

**Predicate Generation**:
- `on(disc, disc)`: All permutations of discs (d1, d2) → on(d1,d2), on(d2,d1)
- `on(disc, peg)`: All disc-peg combinations
- `clear(disc)`: All discs
- `clear(peg)`: All pegs
- `smaller(disc, disc)`: All permutations of discs
- `smaller(peg, disc)`: All peg-disc combinations

### 3. Trajectory Handlers

#### Deterministic: `HanoiImageTrajectoryHandler`

**Components**:
- `HanoiObjectDetector`: Position/size-based detector
- `HanoiFluentClassifier`: Geometric classifier

**Initialization**:
```python
object_name_to_color = {
    ObjectLabel("peg1:peg"): (0.5, 0.5, 0.5),  # Gray
    ObjectLabel("d1:disc"): (0.8, 0.1, 0.1),    # Red (all discs)
    # ...
}
```

#### LLM-Based: `LLMHanoiImageTrajectoryHandler`

**Components**:
- `LLMHanoiObjectDetector`: GPT-4 Vision detector
- `LLMHanoiFluentClassifier`: GPT-4 Vision classifier

**Initialization**:
1. Detect objects in initial state image
2. Extract disc and peg names
3. Set `type_to_objects` mapping for fluent classifier
4. Generate all possible predicates for the specific problem

## Usage Examples

### CLI: Deterministic Hanoi

```bash
# Run with percentage masking (deterministic detection)
python -m src.simulator_cli simple \
  --domain hanoi \
  --problem problem0 \
  --steps 10 \
  --masking percentage \
  --ratio 0.8
```

### CLI: LLM-Based Hanoi

```bash
# Run with LLM detection and masking
python -m src.simulator_cli simple \
  --domain hanoi \
  --problem problem0 \
  --steps 10 \
  --masking llm
```

### Programmatic Usage

```python
from pathlib import Path
from src.trajectory_handlers.hanoi_image_trajectory_handler import HanoiImageTrajectoryHandler
from src.trajectory_handlers.llm_hanoi_trajectory_handler import LLMHanoiImageTrajectoryHandler

# Deterministic approach
handler_det = HanoiImageTrajectoryHandler("PDDLEnvHanoi-v0")
handler_det.init_visual_components()

# LLM-based approach
handler_llm = LLMHanoiImageTrajectoryHandler(
    "PDDLEnvHanoi-v0",
    openai_apikey="your-api-key"
)
handler_llm.init_visual_components(init_state_image_path="path/to/init_state.png")
```

## Key Differences from Blocks Domain

| Aspect | Blocks | Hanoi |
|--------|--------|-------|
| **Objects** | Colored blocks, gripper, table | Colored discs, gray pegs |
| **Rendering** | Different colors per block | All discs same color (red) |
| **Detection** | Color-based (unique colors) | Position/size-based (same color) |
| **Predicates** | on, ontable, holding, clear, handempty | on, clear, smaller (static) |
| **Complexity** | Gripper state tracking | Disc size ordering |

## Advantages of Each Approach

### Deterministic
✅ Fast and consistent
✅ No API costs
✅ Works offline
❌ Requires good image quality
❌ Sensitive to lighting/rendering changes
❌ Limited to predefined colors/shapes

### LLM-Based
✅ Robust to variations in rendering
✅ Can handle unexpected scenarios
✅ Natural language understanding
❌ Requires API access and costs
❌ Slower (network latency)
❌ Non-deterministic (may vary between runs)

## Integration with Simulator

The Hanoi domain is fully integrated into `simulator_cli.py`:

1. **Trajectory Handler Selection**: Automatically selects appropriate handler based on domain and masking strategy
2. **Config Support**: Uses `config.yaml` for domain configuration
3. **Cross-Validation**: Supports both simple and full pipeline modes

**Configuration** (`config.yaml`):
```yaml
domains:
  hanoi:
    domain_name: "PDDLEnvHanoi-v0"
    domain_file: "src/domains/hanoi/hanoi.pddl"
    problems_dir: "src/domains/hanoi/problems"
    problem_prefix: "problem"
```

## Testing

To test the Hanoi implementation:

1. **Test deterministic detector**:
```bash
python -m src.simulator_cli simple --domain hanoi --problem problem0 --steps 5 --masking percentage --ratio 0.5
```

2. **Test LLM detector**:
```bash
python -m src.simulator_cli simple --domain hanoi --problem problem0 --steps 5 --masking llm
```

3. **Test cross-validation**:
```bash
python -m src.simulator_cli full --domain hanoi --problems problem0 problem1 --steps 10 --masking llm
```

## Future Enhancements

Potential improvements for the Hanoi implementation:

1. **Custom Renderer**: Create a custom pddlgym renderer for Hanoi that assigns different colors to each disc (matching the LLM prompt colors)
2. **Size Detection**: Enhance deterministic detector to use actual disc widths for more accurate size identification
3. **Peg Height Analysis**: Use peg height analysis to verify disc stacking
4. **Hybrid Approach**: Combine deterministic detection with LLM verification
5. **Multi-Modal LLM**: Use both vision and text descriptions for better accuracy

## Troubleshooting

**Issue**: Discs not detected correctly
- **Solution**: Check image quality, adjust `color_threshold` in `HanoiObjectDetector`

**Issue**: Incorrect disc ordering
- **Solution**: Verify disc sizes are visually distinct, check `min_area_threshold`

**Issue**: LLM returns incomplete predicates
- **Solution**: Check prompt clarity, increase `max_tokens` in API call

**Issue**: Peg positions wrong
- **Solution**: Ensure pegs are vertically oriented, check `horizontal_threshold`

## Summary

The Hanoi domain implementation provides a complete, well-organized system for object detection and fluent classification with both deterministic and LLM-based approaches. The code follows the same architectural patterns as the Blocks domain, making it easy to maintain and extend.
