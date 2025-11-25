# Hanoi LLM Vision Pipeline - Testing Summary

## Overview

Created a comprehensive test script that validates both components of the LLM-based vision pipeline for the Towers of Hanoi domain.

## Components Tested

### 1. Object Detection
- **Input**: Hanoi puzzle images
- **Output**: Detected objects (discs and pegs) with correct IDs
- **Key Features**:
  - Multi-step reasoning prompt (find regions → merge → assign IDs)
  - Automatic name mapping (p1→peg1, p2→peg2, p3→peg3)
  - Size-based disc ordering (d1=smallest, d3=largest)
  - Position-based peg ordering (peg1=leftmost, peg3=rightmost)

### 2. Fluent Classification
- **Input**: Hanoi puzzle images + detected objects
- **Output**: Grounded predicates with confidence scores
- **Key Features**:
  - Dynamic prompt generation using detected objects
  - Confidence scoring (2=certain, 1=uncertain, 0=false)
  - Predicate types: `on`, `clear`, `smaller`
  - Organized output by predicate type

## Files Created/Modified

### New Files
1. **`src/test_hanoi_object_detector.py`** - Main test script
   - Tests both object detection and fluent classification
   - Configurable number of images
   - Optional verbose mode
   - Can skip fluent classification with `--no-fluents`

2. **`src/TEST_HANOI_DETECTOR_README.md`** - Complete documentation
   - Usage examples
   - Expected output
   - Troubleshooting guide

3. **`src/HANOI_TESTING_SUMMARY.md`** - This file

### Modified Files

1. **`src/object_detection/llm_hanoi_object_detector.py`**
   - Added `detect()` method override to map peg names
   - Updated regex pattern: `r"\b[a-z]\d+:[a-z]+\b"`
   - Added temperature parameter

2. **`src/llms/domains/hanoi/prompts.py`** (user-modified)
   - Enhanced object detection with multi-step reasoning
   - Uses p1, p2, p3 for pegs (mapped to peg1, peg2, peg3 by detector)

## Usage

### Quick Start

```bash
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL
source venv11/bin/activate

# Test full pipeline (object detection + fluent classification)
python src/test_hanoi_object_detector.py

# Test only 3 images
python src/test_hanoi_object_detector.py -n 3

# Skip fluent classification (faster)
python src/test_hanoi_object_detector.py --no-fluents

# Verbose mode
python src/test_hanoi_object_detector.py -v
```

### Expected Behavior

For each image, the script:
1. **Detects objects** using multi-step reasoning prompt
2. **Maps peg names** from p1→peg1, p2→peg2, p3→peg3
3. **Validates counts** (expects 3 discs, 3 pegs)
4. **Injects detected objects** into fluent classifier prompt
5. **Extracts predicates** with confidence scores
6. **Groups and displays** predicates by type

## Integration Pipeline

```
Image
  ↓
┌─────────────────────────────────┐
│ Object Detection (LLM)          │
│ - Find red regions              │
│ - Merge into discs              │
│ - Assign IDs by size            │
│ - Detect pegs left-to-right     │
└─────────────────────────────────┘
  ↓
Map names: p1→peg1, p2→peg2, p3→peg3
  ↓
detected_objects = {
  'disc': ['d1', 'd2', 'd3'],
  'peg': ['peg1', 'peg2', 'peg3']
}
  ↓
Sort object names alphabetically (IMPORTANT!)
  ↓
sorted_type_to_objects = {
  'disc': ['d1', 'd2', 'd3'],  # sorted
  'peg': ['peg1', 'peg2', 'peg3']  # sorted
}
  ↓
┌─────────────────────────────────┐
│ Initialize Fluent Classifier    │
│ classifier = LLMHanoiFluent...( │
│   type_to_objects=sorted_...)   │
│ - MUST provide objects to init! │
│ - Generates prompt with objects │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│ Run Fluent Classification       │
│ fluents = classifier.classify(  │
│   image_path)                   │
│ - Extract all predicates        │
│ - Assign confidence scores      │
└─────────────────────────────────┘
  ↓
fluents_with_confidence = {
  'on(d1:disc,d2:disc)': 2,
  'on(d2:disc,d3:disc)': 2,
  'on(d3:disc,peg1:peg)': 2,
  'clear(d1:disc)': 2,
  'clear(peg2:peg)': 2,
  'clear(peg3:peg)': 2,
  'smaller(d2:disc,d1:disc)': 2,
  ...
}
```

**CRITICAL**: Objects must be sorted alphabetically before injection into the fluent classifier to ensure consistent prompt generation.

## Key Design Decisions

### 1. Proper Data Flow and Formatting
**Critical Requirement**: The fluent classifier requires a properly formatted `type_to_objects` dictionary with sorted object names.

**Data Flow**:
```python
# Step 1: Detector returns objects with mapped names
detected_objects = detector.detect(image_path)
# Returns: {'disc': ['d2', 'd1', 'd3'], 'peg': ['peg1', 'peg3', 'peg2']}  # potentially unsorted

# Step 2: Extract and sort object names alphabetically
sorted_type_to_objects = {
    'disc': sorted(detected_objects.get('disc', [])),  # ['d1', 'd2', 'd3']
    'peg': sorted(detected_objects.get('peg', []))     # ['peg1', 'peg2', 'peg3']
}

# Step 3: Initialize classifier WITH sorted objects (CRITICAL!)
# Cannot initialize classifier without type_to_objects
classifier = LLMHanoiFluentClassifier(
    openai_apikey=api_key,
    type_to_objects=sorted_type_to_objects,  # Required parameter!
    model=model_name
)

# Step 4: Classifier generates prompt with sorted object names
# System prompt will include: "Colored discs: d1, d2, d3" and "Gray pegs: peg1, peg2, peg3"
fluents = classifier.classify(image_path)
```

**Why This Matters**:
1. **Cannot instantiate without objects**: The classifier REQUIRES `type_to_objects` to generate the system prompt
2. **Must be sorted**: Consistent ordering ensures the LLM sees the same prompt structure across different images
3. **One classifier per image**: Create a new classifier instance for each image with its detected objects

### 2. Name Mapping Layer
**Problem**: LLM uses "p1, p2, p3" for brevity, but fluent classifier expects "peg1, peg2, peg3"

**Solution**: Added mapping in `LLMHanoiObjectDetector.detect()`:
```python
if 'peg' in detected_objects:
    mapped_pegs = [f"peg{peg[1:]}" if peg.startswith('p') else peg
                   for peg in detected_objects['peg']]
    detected_objects['peg'] = mapped_pegs
```

### 3. Multi-Step Object Detection
**Problem**: Discs can be partially occluded when stacked

**Solution**: Multi-step reasoning in prompt:
1. Find all red regions
2. Merge overlapping/aligned regions into discs
3. Assign IDs based on width (size)

### 4. Dynamic Prompt Generation
**Problem**: Number of objects varies between images/problems

**Solution**: Initialize a new classifier instance per image with detected objects:
```python
# Create classifier with detected objects
classifier = LLMHanoiFluentClassifier(
    openai_apikey=api_key,
    type_to_objects=sorted_type_to_objects,  # Sorted objects!
    model=model_name
)

# Run classification
fluents = classifier.classify(image_path)
```

**Why New Instance Per Image**: The classifier's system prompt is generated during initialization using the provided objects. Each image may have different objects detected, so we create a fresh classifier instance.

## Testing Strategy

### Test Images
- Uses images from: `src/domains/hanoi/experiments/num_steps=10/`
- Shows different puzzle states (initial, after moves, etc.)
- Validates detection works across various configurations

### Validation Checks
1. **Object Counts**: Compares detected vs. expected (3 discs, 3 pegs)
2. **Name Format**: Ensures correct ID format (d1, peg1, etc.)
3. **Predicate Counts**: Tracks total predicates extracted
4. **Confidence Distribution**: Shows certain/uncertain/false breakdown

## Next Steps

1. **Run the test** to verify prompts work correctly
2. **Analyze results** to identify any systematic errors
3. **Adjust prompts** if needed based on test output
4. **Integrate** into full trajectory generation pipeline

## Benefits

✅ **End-to-End Testing**: Validates complete vision pipeline
✅ **Automated Validation**: Checks object counts and formats
✅ **Clear Output**: Organized by predicate type with confidence
✅ **Flexible**: Can test just detection or full pipeline
✅ **Debuggable**: Verbose mode shows step-by-step progress
✅ **Maintainable**: Clean separation of concerns with name mapping

## Commands Reference

```bash
# Full test (recommended first run)
python src/test_hanoi_object_detector.py -v -n 3

# Quick test (5 images, no verbose)
python src/test_hanoi_object_detector.py

# Only object detection
python src/test_hanoi_object_detector.py --no-fluents

# Test all images
python src/test_hanoi_object_detector.py -n 11
```
