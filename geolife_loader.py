
import pandas as pd

def load_trajectory(file_path):

    df = pd.read_csv(
        file_path,
        skiprows=6,
        header=None
    )

    df.columns = [
        "lat",
        "lon",
        "unused1",
        "altitude",
        "days",
        "date",
        "time"
    ]

    return df[["lat","lon"]]
