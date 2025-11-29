from collections import defaultdict
from pathlib import Path
from time import time
from sklearn.metrics import precision_score, recall_score
import pandas as pd

from src.llms.domains.blocks.consts import objects_to_names
from src.llms.domains.blocks.images.gt_for_testing import create_gt_for_testing
from src.llms.domains.blocks.prompts import full_guidance_system_prompt, object_detection_system_prompt, \
    with_uncertain_confidence_system_prompt
from src.llms.facts_extraction import simulate_predicate_probabilities, extract_facts_once, \
    simulate_relevance_judgement, fill_missing_predicates_with_uncertainty
from src.llms.precision_recall_calculations import evaluate_all_pr_curves
from src.llms.precision_recall_calculations.compute_all import extract_confidence_thresholds_from_pr_values
from src.llms.utils import predicate_extraction_regex, object_detection_regex, \
    predicate_extraction_with_relevance_regex, parse_predicate_relevance, parse_predicate_proba, parse_object_detection


def evaluate_model_on_dataset__probabilites(
        image_files,
        ground_truth: dict,
        system_prompt_text: str,
        model: str = "gpt-4o",
        result_regex: str = predicate_extraction_regex,
        result_parse_func: callable = parse_predicate_relevance,
        temperature: float = 1.3,
        trials: int = 10,
        plot_dir: str = "pr_curves_new"
) -> defaultdict:
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    all_results = defaultdict(lambda: {"scores": [], "labels": []})

    for image_path in image_files:
        start_time = time()
        image_id = image_path.stem
        print(f"Processing {image_id}...")
        gt = ground_truth[image_id]

        probs = simulate_predicate_probabilities(
            image_path, model=model, system_prompt_text=system_prompt_text, result_regex=result_regex,
            result_parse_func=result_parse_func, temperature=temperature, trials=trials
        )

        print(probs)

        all_preds = set(probs) | set(gt)
        for pred in all_preds:
            all_results[pred]["scores"].append(probs.get(pred, 0.0))
            all_results[pred]["labels"].append(gt.get(pred, 0))

        elapsed_time = time() - start_time
        print(f"✅ Processed {image_id} in {elapsed_time:.2f} seconds")

    # === Precision-Recall + AP computation ===
    evaluate_all_pr_curves(all_results, output_dir=plot_dir)

    return all_results


def evaluate_model_on_dataset__relevance(
        image_files,
        ground_truth: dict,
        system_prompt_text: str,
        model: str = "gpt-4o",
        result_regex: str = predicate_extraction_with_relevance_regex,
        result_parse_func: callable = parse_predicate_relevance,
        temperature: float = 1.3,
        plot_dir: str = "relevance_eval"
):
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Accumulate per-predicate predictions and labels
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)

    for image_path in image_files:
        image_id = image_path.stem
        print(f"Processing {image_id}...")

        gt = ground_truth[image_id]  # Dict[predicate] = 0 or 1

        relevance = simulate_relevance_judgement(
            image_path,
            model=model,
            system_prompt_text=system_prompt_text,
            result_regex=result_regex,
            result_parse_func=result_parse_func,
            temperature=temperature
        )

        pred_diff = set(relevance.keys()) ^ set(gt.keys())
        if pred_diff:
            print(f"⚠️ Mismatched predicates for {image_id}: {pred_diff}")
            print(f"in GT only: {set(gt.keys()) - set(relevance.keys())}")
            relevance = fill_missing_predicates_with_uncertainty(relevance, set(gt.keys()))
            print(f"in Relevance only: {set(relevance.keys()) - set(gt.keys())}")

        for predicate in set(gt.keys()) | set(relevance.keys()):
            rel = relevance.get(predicate, 0)
            true = gt.get(predicate, 0)

            all_preds[predicate].append(rel)
            all_labels[predicate].append(true)

    # === Evaluation per predicate ===
    results_with_idk = []
    results_without_idk = []
    for predicate in sorted(set(all_preds) | set(all_labels)):
        preds = all_preds.get(predicate, [])
        labels = all_labels.get(predicate, [])
        assert len(preds) == len(labels), f"Mismatch for predicate {predicate}"
        if not preds:
            print(f"⚠️ No predictions for predicate {predicate}, skipping.")
            continue

        tp_pos = fp_pos = fn_pos_certain = fn_pos_uncertain = 0
        tp_neg = fp_neg = fn_neg_certain = fn_neg_uncertain = 0

        for pred, true in zip(preds, labels):
            # --- Positive predicate evaluation
            if pred == 2 and true == 1:
                tp_pos += 1
            elif pred == 2 and true == 0:
                fp_pos += 1
            # elif pred in (0, 1) and true == 1:
            #     fn_pos += 1
            elif pred == 0 and true == 1:
                fn_pos_certain += 1
            elif pred == 1 and true == 1:
                fn_pos_uncertain += 1


            # --- Negative predicate evaluation
            if pred == 0 and true == 0:
                tp_neg += 1
            elif pred == 0 and true == 1:
                fp_neg += 1
            # elif pred in (1, 2) and true == 0:
            #     fn_neg += 1
            elif pred == 2 and true == 0:
                fn_neg_certain += 1
            elif pred == 1 and true == 0:
                fn_neg_uncertain += 1

        precision_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
        recall_pos_certain = tp_pos / (tp_pos + fn_pos_certain) if (tp_pos + fn_pos_certain) > 0 else 0.0
        recall_pos_uncertain = tp_pos / (tp_pos + fn_pos_certain + fn_pos_uncertain) if (tp_pos + fn_pos_certain + fn_pos_uncertain) > 0 else 0.0

        precision_neg = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
        recall_neg_certain = tp_neg / (tp_neg + fn_neg_certain) if (tp_neg + fn_neg_certain) > 0 else 0.0
        recall_neg_uncertain = tp_neg / (tp_neg + fn_neg_certain + fn_neg_uncertain) if (tp_neg + fn_neg_certain + fn_neg_uncertain) > 0 else 0.0

        results_without_idk.append({
            "predicate": predicate,
            "num_samples": len(preds),
            "pos_precision": precision_pos,
            "pos_recall": recall_pos_certain,
            "neg_precision": precision_neg,
            "neg_recall": recall_neg_certain,
        })

        results_with_idk.append({
            "predicate": predicate,
            "num_samples": len(preds),
            "pos_precision": precision_pos,
            "pos_recall": recall_pos_uncertain,
            "neg_precision": precision_neg,
            "neg_recall": recall_neg_uncertain,
        })

    # === Save all predicate results to CSV - WITH UNCERTAINTY===
    df_with_idk = pd.DataFrame(results_with_idk)
    df_with_idk.sort_values("predicate").to_csv(Path(plot_dir) / "predicate_eval_discrete_with_idk.csv", index=False)

    # === Save grouped predicate results to CSV ===
    # Extract predicate type (function name before the first '(')
    df_with_idk_filtered = df_with_idk[(df_with_idk["pos_precision"] > 0) & (df_with_idk["neg_precision"] > 0)]  # Filter out zero metrics
    df_with_idk_filtered["predicate_type"] = df_with_idk_filtered["predicate"].str.extract(r"^(\w+)\(")
    # Group by predicate type
    with_idk_grouped = df_with_idk_filtered.groupby("predicate_type").agg({
        "pos_precision": "mean",
        "pos_recall": "mean",
        "neg_precision": "mean",
        "neg_recall": "mean",
        "num_samples": "sum"
    }).reset_index()

    # Save per-type averages
    with_idk_grouped.to_csv(Path(plot_dir) / "avg_metrics_per_predicate_type_with_idk.csv", index=False)

    # === Save all predicate results to CSV - WITHOUT UNCERTAINTY===
    df_without_idk = pd.DataFrame(results_without_idk)
    df_without_idk.sort_values("predicate").to_csv(Path(plot_dir) / "predicate_eval_discrete_without_idk.csv", index=False)

    # === Save grouped predicate results to CSV ===
    # Extract predicate type (function name before the first '(')
    df_without_idk_filtered = df_without_idk[
        (df_without_idk["pos_precision"] > 0) & (df_without_idk["neg_precision"] > 0)]  # Filter out zero metrics
    df_without_idk_filtered["predicate_type"] = df_without_idk_filtered["predicate"].str.extract(r"^(\w+)\(")
    # Group by predicate type
    without_idk_grouped = df_without_idk_filtered.groupby("predicate_type").agg({
        "pos_precision": "mean",
        "pos_recall": "mean",
        "neg_precision": "mean",
        "neg_recall": "mean",
        "num_samples": "sum"
    }).reset_index()

    # Save per-type averages
    without_idk_grouped.to_csv(Path(plot_dir) / "avg_metrics_per_predicate_type_without_idk.csv", index=False)
    print(f"\n📁 Evaluation complete. Results saved to: {plot_dir}")
    # return df

if __name__ == "__main__":
    testing_dir = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocks")
    plot_dir = Path(testing_dir / "pr_curves_confidence")
    image_dir = Path(testing_dir / "images")
    image_files = sorted(list(image_dir.glob("*.png")))

    object_names = extract_facts_once(image_dir / "state_0000.png",
                                      model="gpt-4o",
                                      system_prompt_text=object_detection_system_prompt,
                                      result_regex=object_detection_regex,
                                      result_parse_func=parse_object_detection,
                                      temperature=1.0)

    block_names = [o for o in object_names if o.endswith(":block")]
    block_colors = [o.split(":")[0] for o in block_names]
    gripper_name = objects_to_names["gripper"] if next(
        o for o in object_names if o.endswith(":gripper")) else "UNK-GRIPPER"

    ground_truth = create_gt_for_testing(block_names, gripper_name)
    predicate_extraction_system_prompt: str = full_guidance_system_prompt(block_colors)
    # predicate_extraction_system_prompt: str = confidence_system_prompt(block_colors)

    # evaluate_model_on_dataset__probabilities(
    #     image_files=image_files[:5],
    #     ground_truth=ground_truth,
    #     system_prompt_text=predicate_extraction_system_prompt,
    #     model="gpt-4o",
    #     result_regex=predicate_extraction_regex,
    #     result_parse_func=parse_predicate_proba,
    #     temperature=1.0,
    #     trials=10,
    #     plot_dir=plot_dir
    # )

    evaluate_model_on_dataset__relevance(
        image_files=image_files[:],
        ground_truth=ground_truth,
        system_prompt_text=with_uncertain_confidence_system_prompt(block_colors),
        model="gpt-4o",
        result_regex=predicate_extraction_with_relevance_regex,
        result_parse_func=parse_predicate_relevance,
        temperature=1.0,
        plot_dir="relevance_eval"
    )

    df = pd.read_excel(plot_dir / "pr_data_micro.xlsx")
    t_low, t_high = (extract_confidence_thresholds_from_pr_values(df['precision'], df['recall'], df['threshold'],min_precision=0.85))
    print(f"Confidence thresholds: t_low={t_low}, t_high={t_high}")


    # print(extract_facts_once("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/llms/domains/blocksworld/images/state_0000.png",
    #                          model="gpt-4o",
    #                          system_prompt_text=full_guidance_system_prompt,
    #                          result_regex=predicate_extraction_regex,
    #                          temperature=1.0))
