import argparse
from pathlib import Path

from vistautils.parameters import YAMLParametersLoader
from vistanlp_sandbox.utils.classification_tools import (
    YES,
    TRUE_POSITIVES,
    FALSE_POSITIVES,
    FALSE_NEGATIVES,
)

from tabulate import tabulate
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_filename", type=Path)
    args = parser.parse_args()
    params = YAMLParametersLoader().load(args.parameter_filename)

    trained_model_base = params.existing_directory("trained_model_base")
    analysis_output_dir = params.existing_directory("analysis_output_dir")

    event_dfs = []
    for event_type_dir in sorted(p for p in trained_model_base.iterdir() if p.is_dir()):
        event_type = event_type_dir.name
        event_df = pd.read_csv(
            str(event_type_dir / "performance_scores.tsv"), sep="\t", index_col=0
        )
        if YES in event_df.index:
            event_dfs.append(pd.Series(event_df.loc[YES], name=event_type))
        else:
            event_dfs.append(
                pd.Series(index=event_df.columns, name=event_type).fillna(0.0)
            )
    combined_event_df = pd.DataFrame(event_dfs)
    for column_name in (TRUE_POSITIVES, FALSE_POSITIVES, FALSE_NEGATIVES):
        combined_event_df.loc[:, column_name] = combined_event_df.loc[
            :, column_name
        ].astype(int)

    combined_event_df.to_csv(str(analysis_output_dir / "performance_metrics.csv"))
    with open(str(analysis_output_dir / "performance_metrics.md"), "w") as handle:
        handle.write(
            tabulate(
                combined_event_df,
                headers=combined_event_df.columns,
                tablefmt="github",
                showindex=True,
            )
        )
