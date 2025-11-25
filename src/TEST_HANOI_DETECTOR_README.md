# Hanoi Object Detector & Fluent Classifier Test Script

## Overview

This test script (`test_hanoi_object_detector.py`) validates the complete LLM-based vision pipeline for the Towers of Hanoi domain by running it on actual experiment images.

## What It Tests

The script tests both components of the vision pipeline:

### 1. Object Detection (`LLMHanoiObjectDetector`)
- **Disc Detection**: Correctly identifies all discs (d1, d2, d3, etc.)
- **Peg Detection**: Correctly identifies all pegs (peg1, peg2, peg3)
- **Size Ordering**: Discs are ordered by size (d1 = smallest, dn = largest)
- **Position Ordering**: Pegs are ordered left-to-right (peg1 = leftmost)
- **Name Mapping**: Automatically converts LLM output (p1, p2, p3) to expected format (peg1, peg2, peg3)

### 2. Fluent Classification (`LLMHanoiFluentClassifier`)
- **Predicate Extraction**: Extracts all grounded predicates from the image
- **Confidence Scoring**: Assigns confidence levels (certain=2, uncertain=1, false=0)
- **Predicate Types**: Detects `on`, `clear`, and `smaller` predicates
- **Dynamic Prompting**: Uses detected objects to generate domain-specific prompts

## Changes Made to Hanoi Components

### 1. Enhanced Object Detection Prompt (`src/llms/domains/hanoi/prompts.py`)

The prompt now uses a **multi-step reasoning process**:
- **STEP 1**: Find all red regions in the image
- **STEP 2**: Merge regions into physical discs
- **STEP 3**: Assign disc IDs by size (d1 = smallest)
- **STEP 4**: Detect pegs and assign IDs left-to-right (p1, p2, p3)

This structured approach helps the LLM correctly identify discs even when they're partially occluded or stacked.

### 2. Added Name Mapping (`src/object_detection/llm_hanoi_object_detector.py`)

The object detector now automatically converts LLM output to the expected format:
- **LLM Output**: p1, p2, p3 (shorter names)
- **Mapped Output**: peg1, peg2, peg3 (expected by fluent classifier)

```python
def detect(self, image: Union[Path, str], *args, **kwargs) -> Dict[str, List[str]]:
    """Detect objects and map peg names: p1 → peg1, p2 → peg2, etc."""
    detected_objects = super().detect(image, *args, **kwargs)

    # Map peg names
    if 'peg' in detected_objects:
        mapped_pegs = [f"peg{peg[1:]}" if peg.startswith('p') else peg
                       for peg in detected_objects['peg']]
        detected_objects['peg'] = mapped_pegs

    return detected_objects
```

### 3. Updated Regex Pattern

Changed from `r"\b[a-z]+:[a-z]+\b"` to `r"\b[a-z]\d+:[a-z]+\b"` to match:
- `d1:disc`, `d2:disc`, `d3:disc`
- `p1:peg`, `p2:peg`, `p3:peg`

## Usage

### Basic Usage (Full Pipeline)

Test both object detection and fluent classification on 5 images:
```bash
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL
source venv11/bin/activate
python src/test_hanoi_object_detector.py
```

### Test More Images

Test on 10 images:
```bash
python src/test_hanoi_object_detector.py --num-images 10
```

### Object Detection Only

Skip fluent classification (faster, only test object detection):
```bash
python src/test_hanoi_object_detector.py --no-fluents
```

### Verbose Mode

Show detailed step-by-step output:
```bash
python src/test_hanoi_object_detector.py --verbose
```

### Combined Options

Test 3 images in verbose mode:
```bash
python src/test_hanoi_object_detector.py -v -n 3
```

## Sample Output

```
================================================================================
HANOI OBJECT DETECTOR & FLUENT CLASSIFIER TEST
================================================================================
Model: gpt-4o
Test fluent classification: True

Found 5 images to test

================================================================================
IMAGE 1/5: state_0000.png
================================================================================
Path: src/domains/hanoi/experiments/num_steps=10/.../state_0000.png

✓ Object Detection Complete!

Detected Objects:
----------------------------------------
  Discs (3):
    - d1
    - d2
    - d3

  Pegs (3):
    - peg1
    - peg2
    - peg3

Total objects detected: 6

Validation:
  ✓ Discs: 3/3 expected
  ✓ Pegs: 3/3 expected

================================================================================
✓ Fluent Classification Complete!

Detected Predicates (with confidence):
----------------------------------------

  clear predicates (4):
    ✓✓ (certain) clear(d1:disc)
    ✓✓ (certain) clear(peg2:peg)
    ✓✓ (certain) clear(peg3:peg)
    ✗✗ (false) clear(d2:disc)

  on predicates (3):
    ✓✓ (certain) on(d1:disc,d2:disc)
    ✓✓ (certain) on(d2:disc,d3:disc)
    ✓✓ (certain) on(d3:disc,peg1:peg)

  smaller predicates (12):
    ✓✓ (certain) smaller(d2:disc,d1:disc)
    ✓✓ (certain) smaller(d3:disc,d1:disc)
    ✓✓ (certain) smaller(d3:disc,d2:disc)
    ✓✓ (certain) smaller(peg1:peg,d1:disc)
    ✓✓ (certain) smaller(peg1:peg,d2:disc)
    ✓✓ (certain) smaller(peg1:peg,d3:disc)
    ... (and so on for all peg-disc pairs)

Total predicates extracted: 19
  Certain (confidence=2): 18
  Uncertain (confidence=1): 0
  False (confidence=0): 1
```

## Expected Results

For a standard 3-disc Towers of Hanoi problem:
- **Discs**: Should detect 3 discs (d1, d2, d3)
  - d1 = smallest disc
  - d2 = medium disc
  - d3 = largest disc

- **Pegs**: Should detect 3 pegs (peg1, peg2, peg3)
  - peg1 = leftmost peg
  - peg2 = middle peg
  - peg3 = rightmost peg

## Validation

The script automatically validates:
1. **Disc Count**: Compares detected discs vs. expected (3 for standard problem)
2. **Peg Count**: Compares detected pegs vs. expected (3 for standard problem)
3. **Format**: Ensures all objects follow the "name:type" format

## Images Tested

The script uses images from:
```
src/domains/hanoi/experiments/num_steps=10/PDDLEnvHanoi_operator_actions-v0_problem0_temp/
```

Images show different states of the Hanoi puzzle:
- **state_0000.png**: Initial state (all discs on leftmost peg)
- **state_0001.png**: After first move
- **state_0002.png**: After second move
- ... and so on

## Troubleshooting

### API Key Error

**Error**: `❌ Error: Please set your OpenAI API key in config.yaml`

**Solution**: Add your OpenAI API key to `config.yaml`:
```yaml
openai:
  api_key: "sk-..."
```

### No Images Found

**Error**: `❌ Error: No images found in ...`

**Solution**: Make sure the experiment directory exists with PNG images:
```bash
ls -la src/domains/hanoi/experiments/num_steps=10/*/state_*.png
```

### Detection Errors

If detection fails or produces unexpected results:
1. Run with `--verbose` to see detailed output
2. Check the prompt in `src/llms/domains/hanoi/prompts.py`
3. Verify the regex pattern matches the expected format
4. Check API quota and model availability

## Integration with Fluent Classifier

The object detector is designed to work seamlessly with `LLMHanoiFluentClassifier`:

```python
# Step 1: Detect objects
detector = LLMHanoiObjectDetector(openai_apikey, model="gpt-4o")
detected_objects = detector.detect(image_path)
# Returns: {'disc': ['d1', 'd2', 'd3'], 'peg': ['peg1', 'peg2', 'peg3']}

# Step 2: Classify fluents using detected objects
classifier = LLMHanoiFluentClassifier(
    openai_apikey,
    type_to_objects=detected_objects,
    model="gpt-4o"
)
fluents = classifier.classify(image_path)
```

## Files Modified

1. **`src/llms/domains/hanoi/prompts.py`**
   - Updated peg naming: p1/p2/p3 → peg1/peg2/peg3

2. **`src/object_detection/llm_hanoi_object_detector.py`**
   - Added custom regex pattern for digit-containing names
   - Added temperature parameter

3. **`src/test_hanoi_object_detector.py`** (NEW)
   - Test script for validating object detection

## Next Steps

After verifying object detection works correctly:
1. Test the fluent classifier with detected objects
2. Run full trajectory generation with Hanoi domain
3. Compare results with ground truth if available

## Notes

- The object detector uses GPT-4o by default (can be configured in `config.yaml`)
- Detection runs with temperature=1.0 by default for variety
- Images are encoded to base64 before sending to the API
- Each API call processes one image at a time
