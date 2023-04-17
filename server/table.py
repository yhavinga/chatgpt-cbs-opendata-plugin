import io

import cbsodata
from diskcache import Cache

from server.constants import CACHE_DIRECTORY, CACHE_SIZE_LIMIT
from server.pandas_custom_display import pd

cache = Cache(CACHE_DIRECTORY, size_limit=CACHE_SIZE_LIMIT)


def save_dataframe_to_cache(identifier, df):
    cache[identifier] = df.to_csv(index=False)


def load_dataframe_from_cache(identifier):
    csv_str = cache[identifier]
    return pd.read_csv(io.StringIO(csv_str))


def get_table_data(identifier):
    if identifier in cache:
        return load_dataframe_from_cache(identifier)

    data = cbsodata.get_data(identifier)
    df = pd.DataFrame(data)
    save_dataframe_to_cache(identifier, df)
    return df


def table_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = {
        "missing": df.isna().sum(),
        "unique": df.nunique(),
    }

    summary_rows.update(
        {
            stat: df.apply(
                lambda x: x.__getattribute__(stat)()
                if x.dtype.kind in "biufc"
                else None
            )
            for stat in ["mean", "std", "min", "max"]
        }
    )

    summary_rows.update(
        {
            f"{int(q * 100)}%": df.apply(
                lambda x: x.quantile(q) if x.dtype.kind in "biufc" else None
            )
            for q in [0.25, 0.50, 0.75]
        }
    )

    top_5_categories = df.select_dtypes(include=["object", "category"]).apply(
        lambda x: x.value_counts().head(5).to_dict()
    )
    summary_rows["top_5"] = top_5_categories

    return pd.DataFrame(summary_rows)
