from pathlib import Path
import pandas as pd


def load_dataset(data_dir, filename, **kwargs):

    return pd.read_csv(
        Path(data_dir) / filename,
        sep=";",
        **kwargs,
    )


def load_fine(data_dir, filename, **kwargs):

    df = pd.read_csv(
        Path(data_dir) / filename,
        sep=";",
        index_col=[0, 1, 2],
        **kwargs,
    )

    df_reset = df.reset_index()

    df_reset[["coarse", "mid", "fine"]] = (
        df_reset[["coarse", "mid", "fine"]].ffill()
    )

    df = df_reset.set_index(["coarse", "mid", "fine"])

    return df



def aggregate_regions(mid_df, aggregated_codex):

    aggregated_rows = []
    aggregated_codex = aggregated_codex.drop(['Coarse'],axis=1)
    for index, row in aggregated_codex.iterrows():
        target = row[0]  
        source_regions = row[1].split(', ') 
        valid_source_regions = [region for region in source_regions if region in mid_df.index]

        if valid_source_regions:
            aggregated_row = mid_df.loc[valid_source_regions].mean(axis=0)
            
            aggregated_rows.append(aggregated_row)

    aggregated_mid = pd.DataFrame(aggregated_rows)

    aggregated_mid.index = aggregated_codex.iloc[:, 0]

    return aggregated_mid



def read_csv_auto(path: Path) -> pd.DataFrame:

    try:
        data = pd.read_csv(path)

        if data.shape[1] == 1:
            data = pd.read_csv(path, sep=";")

        return data

    except Exception:
        return pd.read_csv(path, sep=";")


import json

def save_processed_recording(
    recording,
    output_dir,
    name,
    metadata,
):

    output_path = output_dir / name

    recording.save(
        folder=output_path,
        format="binary",
        dtype="float32",
        overwrite=True,
        chunk_duration="1s",
        n_jobs=1,
    )

    recording_metadata = {
        **metadata,
        "name": name,
        "sampling_frequency_Hz": float(
            recording.get_sampling_frequency()
        ),
        "n_channels": int(
            recording.get_num_channels()
        ),
        "channel_ids": list(
            map(str, recording.channel_ids)
        ),
    }

    with (
        output_path / "meta.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            recording_metadata,
            file,
            indent=2,
        )

    return output_path