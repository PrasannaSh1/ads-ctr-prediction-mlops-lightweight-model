"""Finds one of the 4 raw CSV files on disk. Downloads it from Kaggle the first time."""
import glob
import os

import kagglehub


def kaggle_csv(file_name, data_dir, kaggle_dataset):
    """Return the local path of a CSV file such as "raw_sample.csv"."""
    # Each file lives in its own folder, for example data/raw_sample/raw_sample.csv
    folder_name = file_name.replace(".csv", "")
    folder = os.path.join(data_dir, folder_name)

    # Look for the file anywhere inside that folder
    pattern = os.path.join(folder, "**", file_name)
    found = []
    for path in glob.glob(pattern, recursive=True):
        if os.path.isfile(path):
            found.append(path)
    if len(found) > 0:
        return found[0]

    # Not on disk yet: download it from Kaggle
    print(f"downloading {file_name} from Kaggle ({kaggle_dataset}) ...")
    path_in_dataset = file_name + "/" + file_name
    return kagglehub.dataset_download(kaggle_dataset, path=path_in_dataset, output_dir=folder)
