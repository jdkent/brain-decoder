import argparse
import itertools
import json
import os
import os.path as op
from glob import glob

import numpy as np
import pandas as pd

from jobs.utils import DEFAULT_MODEL_IDS, DEFAULT_SECTIONS, build_cognitiveatlas, resolve_project_paths

IMG_TO_DOMAIN = {
    "EMOTION": "emotion",
    "GAMBLING": "gambling",
    "LANGUAGE": "language",
    "MOTOR": "motor",
    "RELATIONAL": "relational",
    "SOCIAL": "social",
    "WM": "working memory",
}


def _get_ground_truth_entry(ground_truth, domain):
    if domain in ground_truth:
        return ground_truth[domain]

    normalized_domain = domain.replace(" ", "_")
    if normalized_domain in ground_truth:
        return ground_truth[normalized_domain]

    raise KeyError(domain)


def _recall_at_n(true_lb, pred_lb, n):
    if isinstance(true_lb, int):
        true_lb = [true_lb]

    return len(np.intersect1d(true_lb, pred_lb[:n])) / len(true_lb)


def _get_parser():
    parser = argparse.ArgumentParser(description="Evaluate HCP decoding predictions")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        default=None,
        help="Path to the repository root.",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default=None,
        help="Optional explicit data directory.",
    )
    parser.add_argument(
        "--results_dir",
        dest="results_dir",
        default=None,
        help="Optional explicit results directory.",
    )
    parser.add_argument(
        "--sections",
        dest="sections",
        nargs="+",
        default=list(DEFAULT_SECTIONS),
        help="Text sections to evaluate.",
    )
    parser.add_argument(
        "--model_ids",
        dest="model_ids",
        nargs="+",
        default=list(DEFAULT_MODEL_IDS),
        help="One or more embedding model identifiers to evaluate.",
    )
    parser.add_argument(
        "--sources",
        dest="sources",
        nargs="+",
        default=["cogatlasred", "cogatlas"],
        help="Vocabulary sources to evaluate.",
    )
    parser.add_argument(
        "--categories",
        dest="categories",
        nargs="+",
        default=["task"],
        help="Vocabulary categories to evaluate.",
    )
    parser.add_argument(
        "--sub_categories",
        dest="sub_categories",
        nargs="+",
        default=["combined", "names"],
        help="Vocabulary embedding variants to evaluate.",
    )
    parser.add_argument(
        "--models",
        dest="models",
        nargs="+",
        default=["neurosynth", "gclda", "brainclip"],
        help="Prediction backends expected in the output directory.",
    )
    parser.add_argument(
        "--prediction_dir",
        dest="prediction_dir",
        default=None,
        help="Optional explicit prediction directory. Defaults to results/predictions_hcp_nv.",
    )
    parser.add_argument(
        "--image_dir",
        dest="image_dir",
        default=None,
        help="Optional explicit path to the HCP NeuroVault images.",
    )
    parser.add_argument(
        "--ground_truth_fn",
        dest="ground_truth_fn",
        default=None,
        help="Path to the HCP ground-truth mapping file.",
    )
    parser.add_argument(
        "--output_fn",
        dest="output_fn",
        default=None,
        help="Optional explicit output CSV path.",
    )
    return parser


def main(
    project_dir=None,
    data_dir=None,
    results_dir=None,
    sections=None,
    model_ids=None,
    sources=None,
    categories=None,
    sub_categories=None,
    models=None,
    prediction_dir=None,
    image_dir=None,
    ground_truth_fn=None,
    output_fn=None,
):
    _, data_dir, results_dir = resolve_project_paths(project_dir, data_dir, results_dir)
    sections = list(DEFAULT_SECTIONS) if sections is None else sections
    model_ids = list(DEFAULT_MODEL_IDS) if model_ids is None else model_ids
    sources = ["cogatlasred", "cogatlas"] if sources is None else sources
    categories = ["task"] if categories is None else categories
    sub_categories = ["combined", "names"] if sub_categories is None else sub_categories
    models = ["neurosynth", "gclda", "brainclip"] if models is None else models

    prediction_dir = (
        op.join(results_dir, "predictions_hcp_nv") if prediction_dir is None else op.abspath(prediction_dir)
    )
    image_dir = op.join(data_dir, "hcp", "neurovault") if image_dir is None else op.abspath(image_dir)
    ground_truth_fn = (
        op.join(data_dir, "hcp", "ground_truth.json")
        if ground_truth_fn is None
        else op.abspath(ground_truth_fn)
    )
    output_fn = (
        op.join(results_dir, "eval-hcp-group_results.csv") if output_fn is None else op.abspath(output_fn)
    )

    os.makedirs(prediction_dir, exist_ok=True)
    if not op.exists(ground_truth_fn):
        raise FileNotFoundError(
            f"Ground-truth file not found at {ground_truth_fn}. Pass --ground_truth_fn explicitly."
        )
    images = sorted(glob(op.join(image_dir, "*.nii.gz")))

    with open(ground_truth_fn, "r") as file:
        ground_truth = json.load(file)

    domains = [
        "emotion",
        "gambling",
        "language",
        "motor",
        "relational",
        "social",
        "working memory",
    ]
    subdomains = ["task", "concept", "domain"]

    results_dict = {
        "model": [],
        "task_gclda": [],
        "task_neurosynth": [],
        "task_brainclip": [],
        "concept": [],
        "process": [],
    }

    for section, model_id, source, category, sub_category in itertools.product(
        sections,
        model_ids,
        sources,
        categories,
        sub_categories,
    ):
        model_name = model_id.split("/")[-1]
        reduced = source == "cogatlasred"
        concept_to_process_fn = op.join(data_dir, "cognitive_atlas", "concept_to_process.json")
        cognitiveatlas = build_cognitiveatlas(data_dir, reduced, concept_to_process_fn)

        vocabulary_label = f"vocabulary-{source}_{category}-{sub_category}_embedding-{model_name}_section-{section}"
        results_dict["model"].append(vocabulary_label)

        for model in models:
            if model != "brainclip" and sub_category != "names":
                results_dict[f"task_{model}"].append(np.nan)
                continue

            temp_results = {domain: {subdomain: [] for subdomain in subdomains} for domain in domains}
            for img_fn in images:
                image_name = op.basename(img_fn).split(".")[0]
                task_name = image_name.split("_")[1]
                file_label = f"{task_name}_{vocabulary_label}"

                domain = IMG_TO_DOMAIN[task_name]
                domain_ground_truth = _get_ground_truth_entry(ground_truth, domain)
                task_true_idx = cognitiveatlas.get_task_idx_from_names(domain_ground_truth["task"])

                task_out_fn = f"{file_label}_pred-task_{model}.csv"
                task_prob_df = pd.read_csv(op.join(prediction_dir, task_out_fn))
                task_pred_idx = cognitiveatlas.get_task_idx_from_names(task_prob_df["pred"].values[:5])
                task_recall = _recall_at_n(task_true_idx, task_pred_idx, 4)
                temp_results[domain]["task"].append(task_recall)

                if model != "brainclip":
                    continue

                concept_out_fn = f"{file_label}_pred-concept_{model}.csv"
                process_out_fn = f"{file_label}_pred-process_{model}.csv"

                concept_true_idx = cognitiveatlas.get_concept_idx_from_names(domain_ground_truth["concept"])
                process_true_idx = cognitiveatlas.get_process_idx_from_names(domain_ground_truth["domain"])

                concept_prob_df = pd.read_csv(op.join(prediction_dir, concept_out_fn))
                process_prob_df = pd.read_csv(op.join(prediction_dir, process_out_fn))
                concept_pred_idx = cognitiveatlas.get_concept_idx_from_names(concept_prob_df["pred"].values)
                process_pred_idx = cognitiveatlas.get_process_idx_from_names(process_prob_df["pred"].values)
                concept_recall = _recall_at_n(concept_true_idx, concept_pred_idx, 4)
                process_recall = _recall_at_n(process_true_idx, process_pred_idx, 2)
                temp_results[domain]["concept"].append(concept_recall)
                temp_results[domain]["domain"].append(process_recall)

            mean_task_recalls = np.mean([temp_results[domain]["task"] for domain in domains])
            results_dict[f"task_{model}"].append(mean_task_recalls)

            if model == "brainclip":
                mean_concept_recalls = np.mean([temp_results[domain]["concept"] for domain in domains])
                mean_process_recalls = np.mean([temp_results[domain]["domain"] for domain in domains])
                results_dict["concept"].append(mean_concept_recalls)
                results_dict["process"].append(mean_process_recalls)

    pd.DataFrame(results_dict).to_csv(output_fn, index=False)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
