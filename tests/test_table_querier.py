import pandas as pd
import pytest

from server.table import TableQuerier


@pytest.fixture
def table_querier():
    # Create a sample DataFrame
    data = {"column1": [10, 20, 60, 80], "column2": ["A", "B", "C", "D"]}
    df = pd.DataFrame(data)

    # Initialize the TableQuerier with the sample DataFrame
    return TableQuerier(df)


def test_table_querier(table_querier):
    # Test the translate_query method
    query = "Filter rows where column1 is greater than 50"
    pandas_expression = table_querier.translate_query(query)
    assert (
        pandas_expression == "result = df[df['column1'] > 50]"
    ), "Pandas expression not as expected"

    # Test the query method
    result_df = table_querier.query(query)
    expected_result = pd.DataFrame({"column1": [60, 80], "column2": ["C", "D"]})

    # Reset the index for both DataFrames before comparing them
    result_df.reset_index(drop=True, inplace=True)
    expected_result.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(result_df, expected_result, check_dtype=False)


def test_table_querier(table_querier):
    completion = """I apologize for any offense caused. Here is the code to answer your query:

```
result = df[(df['Landbouwdieren'] == 'Pluimvee (totaal)') & (df['Perioden'] == '2022 december')]['Veestapel_1']
```
"""
    query = table_querier.clean_response(completion)
    assert (
        query
        == "result = df[(df['Landbouwdieren'] == 'Pluimvee (totaal)') & (df['Perioden'] == '2022 december')]['Veestapel_1']"
    )
