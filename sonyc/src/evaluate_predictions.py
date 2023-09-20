import argparse
import json
import os
import oyaml as yaml
import pandas as pd
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
import pdb
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Evaluation script for Urban Sound Tagging task for the DCASE 2019 Challenge.

        See `metrics.py` for more information about the metrics.
        """
    )

    parser.add_argument(
        "prediction_path", type=str, help="Path to prediction CSV file."
    )
    parser.add_argument(
        "annotation_path", type=str, help="Path to dataset annotation CSV file."
    )
    parser.add_argument(
        "yaml_path", type=str, help="Path to dataset taxonomy YAML file."
    )

    args = parser.parse_args()

    with open(args.yaml_path, "r") as f:
        taxonomy = yaml.load(f)

    conf_thres = 0.5
    metrics = {}
    for mode in ["coarse"]:
        metrics[mode] = {}

        predictions_df = pd.read_csv(args.prediction_path)
        annotations_df = pd.read_csv(args.annotation_path)
        res = pd.concat([predictions_df, annotations_df], axis=1, join="inner")

        annotations_df = res[
            ["person", "bicycle", "car", "motorcycle", "siren", "bus", "truck"]
        ]
        predictions_df = res[
            [
                "1_person",
                "2_bicycle",
                "3_car",
                "4_motorcycle",
                "5_siren",
                "6_bus",
                "7_truck",
            ]
        ]

        predictions = np.array(list(predictions_df.itertuples(index=False, name=None)))
        predictions[predictions < conf_thres] = 0
        predictions[predictions >= conf_thres] = 1
        annotations = np.array(list(annotations_df.itertuples(index=False, name=None)))

        f1_per_class = f1_score(predictions, annotations, average=None, zero_division=0)
        precision_per_class = precision_score(
            predictions, annotations, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            predictions, annotations, average=None, zero_division=0
        )

        f1 = f1_score(predictions, annotations, average="macro", zero_division=0)
        precision = precision_score(
            predictions, annotations, average="macro", zero_division=0
        )
        recall = recall_score(
            predictions, annotations, average="macro", zero_division=0
        )

        data = {
            "class": [
                "all",
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "siren",
                "bus",
                "truck",
            ],
            "precision": [precision] + list(precision_per_class),
            "recall": [recall] + list(recall_per_class),
            "f1": [f1] + list(f1_per_class),
        }
        results = pd.DataFrame(data)
        print(results.to_markdown())
        with open(
            os.path.join(os.path.dirname(args.prediction_path), "eval_table.md"), "w"
        ) as fp:
            fp.write(results.to_markdown())
        # df_dict = evaluate(args.prediction_path,
        #                    args.annotation_path,
        #                    args.yaml_path,
        #                    mode)
        # micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
        # macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)

        # # Get index of first threshold that is at least 0.5
        # thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]

        # metrics[mode]["micro_auprc"] = micro_auprc
        # metrics[mode]["micro_f1"] = eval_df["F"][thresh_0pt5_idx]
        # metrics[mode]["macro_auprc"] = macro_auprc
        # metrics[mode]["class_auprc"] = {}

        # print("{} level evaluation:".format(mode.capitalize()))
        # print("======================")
        # print(" * Micro AUPRC:           {}".format(metrics[mode]["micro_auprc"]))
        # print(" * Micro F1-score (@0.5): {}".format(metrics[mode]["micro_f1"]))
        # print(" * Macro AUPRC:           {}".format(metrics[mode]["macro_auprc"]))
        # print(" * Coarse Tag AUPRC:")

        # for coarse_id, auprc in class_auprc.items():
        #     coarse_name = taxonomy["coarse"][int(coarse_id)]
        #     metrics[mode]["class_auprc"][coarse_name] = auprc
        #     print("      - {}: {}".format(coarse_name, auprc))

        # prediction_fname = \
        # os.path.splitext(os.path.basename(args.prediction_path))[0]
        # eval_fname = "evaluation_{}.json".format(prediction_fname)
        # eval_path = os.path.join(os.path.dirname(args.prediction_path),
        #                          eval_fname)
        # with open(eval_path, 'w') as f:
        #     json.dump(metrics, f)
