import base64

predicate_extraction_regex = r"\b[a-z]+\(.*?\)"
object_detection_regex = r"\b[a-z]+:[a-z]+\b" # Matches "color:type" format
predicate_extraction_with_relevance_regex = r"([a-z]+\(.*?\))\s*:\s*([012])"


# TODO: these parsing functions should probably moved to live together with the prompts they parse, also attach to regexes in a configuration
def parse_object_detection(obj_detect_fact: str) -> str:
    return obj_detect_fact.replace(" ", "")  # remove spaces guardedly added by the LLM


def parse_predicate_proba(predicate_fact: str) -> str:
    return predicate_fact.replace(" ", "")  # remove spaces guardedly added by the LLM


def parse_predicate_relevance(predicate_fact: tuple[str, int]) -> tuple[str, int]:
    return predicate_fact[0].replace(" ", ""), int(predicate_fact[1])


def state_key_to_index(state_key: str) -> int:
    # state_key is in the form "state_<state_index>"
    return int(state_key.split('_')[-1])


def extract_predicate_type(p):
    return p.split("(")[0]


def encode_image(image_path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
