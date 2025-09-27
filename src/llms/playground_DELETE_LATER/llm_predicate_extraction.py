import base64
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter, defaultdict
from openai import OpenAI

openai_apikey = "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"
openai_client = OpenAI(api_key=openai_apikey)

def simulate_and_analyze_predicates(
    openai_client,
    image_path,
    system_prompt,
    models=["gpt-4o", "gpt-4o-mini"],
    temperature_values=[1.0, 1.3, 1.5],
    n_trials=10,
    n_repeats=5,
    raw_output_path="raw_predicate_probs.csv",
    summary_output_path="predicate_reliability_analysis_with_stats.xlsx",
    plot_output_path="predicate_cv_vs_mean_plot.png"
):
    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_predicates_once(image_path, model, system_prompt, temperature=1.3):
        base64_image = encode_image(image_path)
        system_prompt = system_prompt

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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000
        )
        text = response.choices[0].message.content.strip()
        return set(re.findall(r"\b[a-z]+\(.*?\)", text))

    def simulate_predicate_probabilities(image_path, model, system_prompt, trials=10, temperature=1.3):
        predicate_counts = Counter()
        for _ in range(trials):
            predicates = extract_predicates_once(image_path, model, system_prompt, temperature=temperature)
            predicate_counts.update(predicates)
        return {p: round(predicate_counts[p] / trials, 2) for p in predicate_counts}

    all_predicates_set = set()
    data = []
    for model in models:
        for temp in temperature_values:
            all_preds_in_run = set()
            rep_data = [defaultdict(lambda: 0.0) for _ in range(n_repeats)]
            for rep in range(n_repeats):
                print(f"Running model={model}, temp={temp}, rep={rep}")
                start = time.time()
                probs = simulate_predicate_probabilities(image_path, model, system_prompt, trials=n_trials, temperature=temp)
                duration = time.time() - start
                all_preds_in_run.update(probs.keys())
                for pred in probs:
                    rep_data[rep][pred] = probs[pred]
                print(f"Completed rep {rep} in {duration:.2f}s")

                if model in ["gpt-4o-mini", "gpt-4.1-nano"]:
                    time.sleep(30)

            all_predicates_set.update(all_preds_in_run)
            for pred in all_preds_in_run:
                for rep in range(n_repeats):
                    data.append({
                        "model": model,
                        "temperature": temp,
                        "predicate": pred,
                        "pos_proba": rep_data[rep].get(pred, 0.0),
                        "trials": n_trials,
                        "repeat": rep
                    })

    df = pd.DataFrame(data)
    df.to_csv(raw_output_path, index=False)

    grouped = defaultdict(lambda: {
        "model": None,
        "temperature": None,
        "predicate": None,
        "pos_probas": [0.0] * n_repeats
    })

    for row in data:
        key = (row["model"], row["temperature"], row["predicate"])
        grouped[key]["model"] = row["model"]
        grouped[key]["temperature"] = row["temperature"]
        grouped[key]["predicate"] = row["predicate"]
        grouped[key]["pos_probas"][row["repeat"]] = row["pos_proba"]

    structured_data = []
    for (model, temp, pred), values in grouped.items():
        probas = values["pos_probas"]
        row = {
            "model": model,
            "temperature": temp,
            "predicate": pred,
            **{f"pos_proba_{i}": probas[i] for i in range(n_repeats)},
            "mean_pos_proba": round(np.mean(probas), 3),
            "std_pos_proba": round(np.std(probas), 3),
        }
        structured_data.append(row)

    summary_df = pd.DataFrame(structured_data)
    summary_df["cv"] = summary_df["std_pos_proba"] / summary_df["mean_pos_proba"].replace(0, np.nan)
    summary_df["ci_95_width"] = 1.96 * summary_df["std_pos_proba"] / np.sqrt(n_repeats)
    summary_df["reliability"] = pd.cut(
        summary_df["std_pos_proba"],
        bins=[-0.01, 0.05, 0.15, 0.3, 1.0],
        labels=["robust", "acceptable", "shaky", "unstable"]
    )

    (summary_df.to_excel(summary_output_path, index=False))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=summary_df,
        x="mean_pos_proba",
        y="cv",
        hue="reliability",
        style="reliability",
        s=100
    )
    plt.title("Predicate Consistency: Mean vs CV")
    plt.xlabel("Mean Positive Probability")
    plt.ylabel("Coefficient of Variation (CV)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output_path)

    print(f"\n✅ Summary saved to: {summary_output_path}")
    print(f"📊 Plot saved to: {plot_output_path}")


if __name__ == "__main__":
    simulate_and_analyze_predicates(
        openai_client=openai_client,
        image_path="/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/fluent_classification/images/state_0003.png",  # 👈 Update with your actual image path
        system_prompt=(
            "You are a visual reasoning agent for a robotic planning system. "
            "Given an image, consisted of the following objects: "
            "1. gray-colored gripper (type=gripper), "
            "2. brown-colored table (type=table), "
            "3. colored blocks: red, blue, green, cyan (type=block). "
            "Extract grounded binary predicates in the following forms:\n"
            "- on(x-block, y-block)\n"
            "- ontable(x-block)\n"
            "- handfree(gripper-gripper)\n"
            "- handful(gripper-gripper)\n"
            "- holding(x-block, gripper-gripper)\n"
            "- clear(x-block)\n\n"
            "Only use defined objects with proper types for grounding. Return one predicate per line."
        ),
        # models=["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"],      # You can modify this list
        models=["gpt-4o"],
        temperature_values=[1.0],     # Or add more variations
        n_trials=10,                            # Number of samples per repetition
        n_repeats=1,                            # Number of repetitions
        raw_output_path="raw_predicate_probs.csv",
        summary_output_path="predicate_reliability_analysis_with_stats__no-pred-guidance.xlsx",
        plot_output_path="pred-no-guidance.png"
    )