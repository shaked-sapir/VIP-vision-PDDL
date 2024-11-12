"""
This approach is limited in its "out of the box" functionality as it is not trained on objects
related to our tasks.

I leave this code untouched here (from GPT) for possible future fine-tune trials,
so if it works we can lift this model for multiple domains instead of "tailoring" object detectors
and predicate detectors for each domain separately.
"""

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Load the model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load and preprocess the image
image_path = "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/blocks_images/state_0007.png"
image = Image.open(image_path).convert("RGB")

# Prepare inputs for the model
inputs = processor(image, "how many blocks are in the image and what are their colors?", return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer = processor.decode(outputs.logits.argmax(-1).item())

print(f"Answer: {answer}")

# Prepare inputs for the model
inputs = processor(image, "Is the green block on top of the red block?", return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer = processor.decode(outputs.logits.argmax(-1).item())

print(f"Answer: {answer}")


inputs = processor(image, "Is the green block on top of the blue block?", return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer = processor.decode(outputs.logits.argmax(-1).item())

print(f"Answer: {answer}")

inputs = processor(image, "Are there blocks in the image?", return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer = processor.decode(outputs.logits.argmax(-1).item())

print(f"Answer: {answer}")

inputs = processor(image, "Is there a green block in the image?", return_tensors="pt")

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer = processor.decode(outputs.logits.argmax(-1).item())

print(f"Answer: {answer}")