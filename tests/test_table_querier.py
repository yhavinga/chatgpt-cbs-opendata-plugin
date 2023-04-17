import pandas as pd
import pytest

from server.table_querier import CBSTableQuerier


@pytest.fixture
def table_querier():
    # Create a sample DataFrame
    data = {"column1": [10, 20, 60, 80], "column2": ["A", "B", "C", "D"]}
    df = pd.DataFrame(data)

    # Initialize the TableQuerier with the sample DataFrame
    return CBSTableQuerier(df)


# def test_table_querier_execute(table_querier, mock_openai):
def test_table_querier_execute(table_querier):
    # Test the translate_query method
    query = "Filter rows where column1 is greater than 50"
    pandas_expression = table_querier.translate_query(query)
    assert (
        pandas_expression == "result = df[df['column1'] > 50]"
    ), "Pandas expression not as expected"

    # Test the query method
    pandas_query, result_df = table_querier.query(query)
    expected_result = pd.DataFrame({"column1": [60, 80], "column2": ["C", "D"]})

    # Reset the index for both DataFrames before comparing them
    result_df.reset_index(drop=True, inplace=True)
    expected_result.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(result_df, expected_result, check_dtype=False)


def test_get_code_from_completion(table_querier):
    completion = """I apologize for any offense caused. Here is the code to answer your query:

```
result = df[(df['Landbouwdieren'] == 'Pluimvee (totaal)') & (df['Perioden'] == '2022 december')]['Veestapel_1']
```
"""
    query = table_querier.get_code_from_completion(completion)
    assert (
        query
        == "result = df[(df['Landbouwdieren'] == 'Pluimvee (totaal)') & (df['Perioden'] == '2022 december')]['Veestapel_1']"
    )


# def test_query_table_data_error_82634NED(table_querier):
#     prompt = """df.head(n=3): ```   ID Persoonskenmerken            Cijfersoort  Perioden  ScoreGeluk_1  Ongelukkig_2  NietGelukkigNietOngelukkig_3  Gelukkig_4
#         ScoreTevredenheidMetHetLeven_5  Ontevreden_6  NietTevredenNietOntevreden_7  Tevreden_8  ScoreTevredenheidOpleidingskansen_9  Ontevreden_10
#         NietTevredenNietOntevreden_11  Tevreden_12  ScoreTevredenheidMetWerk_13  Ontevreden_14  NietTevredenNietOntevreden_15  Tevreden_16  ScoreTevredenheidMetReistijd_17
#         Ontevreden_18  NietTevredenNietOntevreden_19  Tevreden_20  ScoreTevredenheidDagelijkseBezigheden_21  Ontevreden_22  NietTevredenNietOntevreden_23  Tevreden_24
#         ScoreTevredenheidMetLichGezondheid_25  Ontevreden_26  NietTevredenNietOntevreden_27  Tevreden_28  ScoreTevredenheidPsychischeGezondheid_29  Ontevreden_30
#         NietTevredenNietOntevreden_31  Tevreden_32  ScoreTevredenheidMetGewicht_33  Ontevreden_34  NietTevredenNietOntevreden_35  Tevreden_36  ScoreTevredenheidFinancieleSituatie_37
#         Ontevreden_38  NietTevredenNietOntevreden_39  Tevreden_40  ScoreZorgenOverFinancieleToekomst_41  GeenZorgen_42  WeinigZorgen_43  VeelZorgen_44  ScoreTevredenheidMetWoning_45
#         Ontevreden_46  NietTevredenNietOntevreden_47  Tevreden_48  ScoreTevredenheidMetWoonomgeving_49  Ontevreden_50  NietTevredenNietOntevreden_51  Tevreden_52  ScoreOnveiligheidsgevoelens_53
#         Veilig_54  NietVeiligNietOnveilig_55  Onveilig_56  ScoreTevredenheidMetSociaalLeven_57  Ontevreden_58  NietTevredenNietOntevreden_59  Tevreden_60  ScoreTevredenhHoeveelheidVrijeTijd_61
#         Ontevreden_62  NietTevredenNietOntevreden_63  Tevreden_64  AandeelMetVertrouwen_65 0   0   Totaal personen  Gemiddelde/Percentage
#     2013           7.7           2.5                          10.0        87.5                             7.5           3.4                          13.0        83.6
#     7.5            5.2                           15.8         79.0                          7.7            3.3                           12.5         84.2
#     8.1            3.9                           11.6         84.4                                       7.2            6.3                           18.2         75.5
#     7.1            9.7                           19.9         70.5                                       7.8            4.7                           10.4         85.0
#     6.9           11.9                           22.5         65.6                                     6.9            9.3                           22.4         68.3
#     5.1           39.8             27.9           32.2                            8.0            3.9                            8.1         88.0
#     7.9            3.4                           10.3         86.3                             3.5       69.2                       16.9         13.9
#     7.7            3.8                           12.7         83.5                                    7.4            7.3                           17.0         75.7
#      58.1 1   1   Totaal personen  Gemiddelde/Percentage      2014           7.7           2.4                           9.7        87.9
#      7.6           3.3                          12.1        84.6                                  7.5            5.0                           15.4         79.6
#      7.6            3.6                           13.3         83.1                              8.1            4.9                           11.2         83.8
#      7.3            6.0                           16.9         77.1                                    7.1            9.4                           19.2         71.4
#      7.8            4.4                           10.6         85.0                             6.9           11.7                           23.2         65.2
#      7.0            8.9                           20.1         70.9                                   4.9           44.6             25.7           29.7                            8.0
#      3.8                            8.4         87.9                                  7.8            3.9                           10.7         85.4                             3.5       70.1
#      16.1         13.8                                  7.6            4.2                           12.3         83.5                                    7.4            7.5
#      17.2         75.3                     57.7 2   2   Totaal personen  Gemiddelde/Percentage      2015           7.7           2.8                           9.8        87.4
#      7.5           3.9                          12.2        83.9                                  7.5            5.4                           15.0         79.7                          7.7
#      3.1                           13.1         83.8                              8.1            4.6                           11.9         83.4                                       7.2
#      5.8                           16.9         77.3                                    7.0           10.9                           19.9         69.3                                       7.8
#      5.1                           11.3         83.6                             6.9           12.6                           21.8         65.5                                     7.0
#      8.6                           20.9         70.5                                   4.9           45.2             25.5           29.4                            8.0            3.5
#      9.6         86.9                                  7.9            3.8                           10.3         85.9                             3.5       69.2                       16.7
#      14.1                                  7.6            4.3                           12.5         83.2                                    7.3            7.7                           17.9
#      74.4                     59.5```
#      query: ```percentage of people aged 15-19 who reported poor mental health or depression in the Netherlands from 2013 to 2021```
# NB! the data above is an example so you know the format. Do not come to the conclusion that the query cannot be answered
# because the data is incomplete. Your answer will run on the complete dataframe.
# Your task it to write one or more python pandas instructions on this dataframe to answer the user query.
# Think step by step how to massage the dataframe to match the user query.
# Return one or more python code lines with pandas dataframe operations to answer the query and assign the result to the 'result' variable.
# The result of the code must be a pandas dataframe, not a list or numpy array.
# """
#
#     prompt = re.sub(r"\s+", " ", prompt)
#     messages = [
#         {
#             "role": "system",
#             "content": """Answer as a highly experienced python pandas programmer without attitude.
# Your task is to convert natural language queries to python pandas code.
# Assume the user gives example rows and your response will run on the complete data.
# Make the best possible assumption mapping the query to the columns.""".replace(
#                 "\n", " "
#             ),
#         },
#         {"role": "user", "content": prompt},
#     ]
#     completion = get_chat_completion(messages)
#     query = table_querier.get_code_from_completion(completion)
#
#     print(query)
