import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

from vistanlp_sandbox.utils.classification_tools import (
    _compute_prf,
    TRUE_POSITIVES,
    FALSE_POSITIVES,
    FALSE_NEGATIVES,
    PRECISION,
    RECALL,
    F1,
)
from gaia_event_extraction.drivers.score_model_for_ace_data import ALL_EVENTS_COMBINED

from tabulate import tabulate
import numpy as np
import pandas as pd


def rescore(df: pd.DataFrame, *, sum_row_name: Optional[str] = None) -> pd.DataFrame:
    s = deepcopy(df)
    if sum_row_name is not None:
        s.loc[sum_row_name] = s.sum()
    return _compute_prf(s.drop(columns=[PRECISION, RECALL, F1]).to_dict(orient="index"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trained_model_base", type=Path)
    parser.add_argument("analysis_output_dir", type=Path)
    args = parser.parse_args()
    trained_model_base = args.trained_model_base.resolve()
    analysis_output_dir = args.analysis_output_dir.resolve()

    combined_event_df = pd.read_csv(
        str(trained_model_base / "performance_scores.tsv"), sep="\t", index_col=0
    )
    for column_name in (TRUE_POSITIVES, FALSE_POSITIVES, FALSE_NEGATIVES):
        combined_event_df.loc[:, column_name] = combined_event_df.loc[
            :, column_name
        ].astype(int)
    # just in case
    # combined_event_df.index = combined_event_df.index.map(
    #     lambda x: x.replace("::", ".")
    # )
    combined_event_df.drop(np.nan, inplace=True)
    combined_event_df.sort_index(inplace=True)
    combined_event_df = rescore(combined_event_df, sum_row_name=ALL_EVENTS_COMBINED)
    print(combined_event_df)
    combined_event_df.to_csv(str(analysis_output_dir / "performance_metrics.csv"))
    with open(str(analysis_output_dir / "performance_metrics.md"), "w") as handle:
        handle.write(
            tabulate(
                combined_event_df,
                headers=combined_event_df.columns,
                tablefmt="github",
                showindex=True,
                floatfmt=".2f",
            )
        )
