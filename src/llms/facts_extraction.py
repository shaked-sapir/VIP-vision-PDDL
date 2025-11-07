import re
from collections import Counter
from pathlib import Path

from openai import OpenAI

from src.llms.utils import encode_image


def print_RGB(image_path: Path):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path)
    img = img.convert("RGB")
    img_array = np.array(img)

    # Get the dimensions of the image
    height, width, channels = img_array.shape

    # Print RGB values for each pixel
    for y in range(height):
        for x in range(width):
            r, g, b = img_array[y, x]
            print(f"Pixel at ({x}, {y}): R={r}, G={g}, B={b}")

def extract_facts_once(image_path: Path, model: str, system_prompt_text: str, result_regex: str,
                       result_parse_func: callable, temperature=1.3):
    base64_image: str = encode_image(image_path)
    user_prompt = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        },
        {
            "type": "text",
            "text": "Extract all predicates as described above. Return one predicate per line."
        }
    ]

    response = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt}
        ],
        # max_tokens=3000
    )
    response_text: str = response.choices[0].message.content.strip()
    facts: list[str] = re.findall(result_regex, response_text)
    return set([result_parse_func(fact) for fact in facts])  # remove spaces guardedly added by the LLM


def simulate_predicate_probabilities(image_path: Path, model: str, system_prompt_text: str, result_regex: str,
                                     result_parse_func: callable, temperature: float, trials: int = 10):
    predicate_counts = Counter()
    for _ in range(trials):
        predicates = extract_facts_once(image_path, model, system_prompt_text, result_regex, result_parse_func,
                                        temperature)
        predicate_counts.update(predicates)
    return {p: predicate_counts[p] / trials for p in predicate_counts}


def simulate_relevance_judgement(image_path: Path, model: str, system_prompt_text: str, result_regex: str,
                                 result_parse_func: callable, temperature: float):
    predicates = extract_facts_once(image_path, model, system_prompt_text, result_regex, result_parse_func, temperature)
    return {p: rel for p, rel in predicates}


def fill_missing_predicates_with_uncertainty(relevance_dict: dict, all_possible_preds: set[str],
                                             uncertainty_label: int = 1) -> dict:
    """
    Fills in any missing predicates with a default relevance score of 1 (uncertain).
    """
    return {pred: relevance_dict.get(pred, uncertainty_label) for pred in all_possible_preds}
