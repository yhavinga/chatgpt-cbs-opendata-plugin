import io
import json
import os
import re
from typing import Union

import cbsodata
import duckdb
import faiss
import numpy as np
import pandas as pd
from diskcache import Cache
from pandas import Index, Series
from RestrictedPython import compile_restricted
from RestrictedPython.Eval import (default_guarded_getattr,
                                   default_guarded_getitem,
                                   default_guarded_getiter)
from RestrictedPython.Guards import (guarded_iter_unpack_sequence,
                                     guarded_unpack_sequence)

from services.openai import get_chat_completion, get_embeddings

TABLE_LIST = "table_list.csv"
EMBEDDINGS = "embeddings.jsonl"

cache_directory = "cache_directory"
cache_size_limit = 5 * 1024 * 1024 * 1024  # 5GB
cache = Cache(cache_directory, size_limit=cache_size_limit)

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 20000)
pd.set_option("display.expand_frame_repr", False)


"""
{'Updated': '2021-11-01T02:00:00', 'ID': 0, 'Identifier': '84669NED', 'Title': "Banen van werknemers; internationaliseringskenmerken van bedrijven, '10-'18", 'ShortTitle': 'Banen; internationalisering bedrijven', 'ShortDescription': '\nDeze tabel bevat informatie over werknemersbanen in Nederland. Het aantal banen wordt uitgesplitst naar diverse achtergrondkenmerken van de werknemers en de bedrijven waarvoor zij in dienst zijn. Bij de bedrijven wordt met name onderscheid gemaakt tussen bedrijven onder Nederlandse zeggenschap en bedrijven onder buitenlandse zeggenschap, en de internationale handel van het bedrijf. Werknemerskenmerken zijn onder meer opleidingsniveau en nationaliteit. Daarbij wordt de top vijf van meest voorkomende nationaliteiten bij bedrijven onder buitenlandse zeggenschap gerapporteerd. Het opleidingsniveau wordt gegeven bij alle selecties (uitgezonderd buitenlandse nationaliteiten) met tenminste duizend directe observaties.\n\nGegevens beschikbaar van 2010 tot en met 2018.\n\nStatus van de cijfers:\nDe cijfers in deze tabel zijn voorlopig.\n\nWijzigingen per 1 november 2021:\nGeen, deze tabel is stopgezet.\n\nWanneer komen er nieuwe cijfers?\nNiet meer van toepassing.\n', 'Summary': 'werknemersbanen, voltijdequivalenten, arbeidsvolume, loon\nInternationale handel in goederen & diensten, multinationals, zeggenschap', 'Modified': '2020-12-14T02:00:00', 'MetaDataModified': '2021-11-01T02:00:00', 'ReasonDelivery': 'Stopgezet', 'ExplanatoryText': '', 'OutputStatus': 'Gediscontinueerd', 'Source': 'CBS.', 'Language': 'nl', 'Catalog': 'CBS', 'Frequency': 'Stopgezet', 'Period': '2010 t/m 2018', 'SummaryAndLinks': 'werknemersbanen, voltijdequivalenten, arbeidsvolume, loon<br />Internationale handel in goederen & diensten, multinationals, zeggenschap<br /><a href="http://opendata.cbs.nl/ODataApi/OData/84669NED">http://opendata.cbs.nl/ODataApi/OData/84669NED</a><br /><a href="http://opendata.cbs.nl/ODataFeed/OData/84669NED">http://opendata.cbs.nl/ODataFeed/OData/84669NED</a>', 'ApiUrl': 'https://opendata.cbs.nl/ODataApi/OData/84669NED', 'FeedUrl': 'https://opendata.cbs.nl/ODataFeed/OData/84669NED', 'DefaultPresentation': 'ts=1606984891185&graphtype=Table&r=InternationaliseringskenmerkenBedrijf&k=Topics&t=BedrijfstakkenBranchesSBI2008,Bedrijfsgrootte,KenmerkenBaanEnWerknemer,Perioden', 'DefaultSelection': "$filter=((BedrijfstakkenBranchesSBI2008 eq 'T001081')) and ((Bedrijfsgrootte eq 'T001098')) and ((KenmerkenBaanEnWerknemer eq 'T001025')) and ((Perioden eq '2018JJ00'))&$select=BedrijfstakkenBranchesSBI2008, Bedrijfsgrootte, InternationaliseringskenmerkenBedrijf, KenmerkenBaanEnWerknemer, Perioden, Banen_1, Arbeidsvolume_2, Uurloon_3, PerBaanPerWeekExclusiefOverwerk_5", 'GraphTypes': 'Table,Bar,Line', 'RecordCount': 304128, 'ColumnCount': 10, 'SearchPriority': '1'}
{'ApiUrl': 'https://opendata.cbs.nl/ODataApi/OData/84669NED',
 'Catalog': 'CBS',
 'ColumnCount': 10,
 'DefaultPresentation': 'ts=1606984891185&graphtype=Table&r=InternationaliseringskenmerkenBedrijf&k=Topics&t=BedrijfstakkenBranchesSBI2008,Bedrijfsgrootte,KenmerkenBaanEnWerknemer,Perioden',
 'DefaultSelection': "$filter=((BedrijfstakkenBranchesSBI2008 eq 'T001081')) "
                     "and ((Bedrijfsgrootte eq 'T001098')) and "
                     "((KenmerkenBaanEnWerknemer eq 'T001025')) and ((Perioden "
                     "eq '2018JJ00'))&$select=BedrijfstakkenBranchesSBI2008, "
                     'Bedrijfsgrootte, InternationaliseringskenmerkenBedrijf, '
                     'KenmerkenBaanEnWerknemer, Perioden, Banen_1, '
                     'Arbeidsvolume_2, Uurloon_3, '
                     'PerBaanPerWeekExclusiefOverwerk_5',
 'ExplanatoryText': '',
 'FeedUrl': 'https://opendata.cbs.nl/ODataFeed/OData/84669NED',
 'Frequency': 'Stopgezet',
 'GraphTypes': 'Table,Bar,Line',
 'ID': 0,
 'Identifier': '84669NED',
 'Language': 'nl',
 'MetaDataModified': '2021-11-01T02:00:00',
 'Modified': '2020-12-14T02:00:00',
 'OutputStatus': 'Gediscontinueerd',
 'Period': '2010 t/m 2018',
 'ReasonDelivery': 'Stopgezet',
 'RecordCount': 304128,
 'SearchPriority': '1',
 'ShortDescription': '\n'
                     'Deze tabel bevat informatie over werknemersbanen in '
                     'Nederland. Het aantal banen wordt uitgesplitst naar '
                     'diverse achtergrondkenmerken van de werknemers en de '
                     'bedrijven waarvoor zij in dienst zijn. Bij de bedrijven '
                     'wordt met name onderscheid gemaakt tussen bedrijven '
                     'onder Nederlandse zeggenschap en bedrijven onder '
                     'buitenlandse zeggenschap, en de internationale handel '
                     'van het bedrijf. Werknemerskenmerken zijn onder meer '
                     'opleidingsniveau en nationaliteit. Daarbij wordt de top '
                     'vijf van meest voorkomende nationaliteiten bij bedrijven '
                     'onder buitenlandse zeggenschap gerapporteerd. Het '
                     'opleidingsniveau wordt gegeven bij alle selecties '
                     '(uitgezonderd buitenlandse nationaliteiten) met '
                     'tenminste duizend directe observaties.\n'
                     '\n'
                     'Gegevens beschikbaar van 2010 tot en met 2018.\n'
                     '\n'
                     'Status van de cijfers:\n'
                     'De cijfers in deze tabel zijn voorlopig.\n'
                     '\n'
                     'Wijzigingen per 1 november 2021:\n'
                     'Geen, deze tabel is stopgezet.\n'
                     '\n'
                     'Wanneer komen er nieuwe cijfers?\n'
                     'Niet meer van toepassing.\n',
 'ShortTitle': 'Banen; internationalisering bedrijven',
 'Source': 'CBS.',
 'Summary': 'werknemersbanen, voltijdequivalenten, arbeidsvolume, loon\n'
            'Internationale handel in goederen & diensten, multinationals, '
            'zeggenschap',
 'SummaryAndLinks': 'werknemersbanen, voltijdequivalenten, arbeidsvolume, '
                    'loon<br />Internationale handel in goederen & diensten, '
                    'multinationals, zeggenschap<br /><a '
                    'href="http://opendata.cbs.nl/ODataApi/OData/84669NED">http://opendata.cbs.nl/ODataApi/OData/84669NED</a><br '
                    '/><a '
                    'href="http://opendata.cbs.nl/ODataFeed/OData/84669NED">http://opendata.cbs.nl/ODataFeed/OData/84669NED</a>',
 'Title': 'Banen van werknemers; internationaliseringskenmerken van bedrijven, '
          "'10-'18",
 'Updated': '2021-11-01T02:00:00'}
"""


def get_table_list():
    # if file exists, load it, else get the table list.
    if os.path.exists(TABLE_LIST):
        df = pd.read_csv(TABLE_LIST)
    else:
        tables = cbsodata.get_table_list()
        df = pd.DataFrame(tables)
        # filter the table list to ReasonDelivery != Stopgezet
        df = df[df.ReasonDelivery != "Stopgezet"]
        save_table_list(df)
    return df


def get_table_summary(identifier):
    df = get_table_list()
    # get the Summary column for the table with the given identifier
    return df[df.Identifier == identifier].Summary.values[0]


def save_dataframe_to_cache(identifier, df):
    cache[identifier] = df.to_csv(index=False)


def load_dataframe_from_cache(identifier):
    csv_str = cache[identifier]
    return pd.read_csv(io.StringIO(csv_str))


def get_table_data(identifier):
    if identifier in cache:
        return load_dataframe_from_cache(identifier)

    data = cbsodata.get_data(identifier)

    # convert to pandas dataframe
    df = pd.DataFrame(data)

    # Save the DataFrame to the cache
    save_dataframe_to_cache(identifier, df)

    return df


def save_table_list(df: pd.DataFrame):
    df.to_csv(TABLE_LIST, index=False)


def get_table_embeddings() -> pd.DataFrame:
    if not os.path.exists(EMBEDDINGS):
        df = get_table_list()
        embed_table(df)
    return pd.read_json(EMBEDDINGS, lines=True)


def embed_table(df: pd.DataFrame):
    batch_size = 32
    embeddings = []

    texts = []
    identifiers = []
    for idx, r in df.iterrows():
        identifier = r.Identifier
        text = " ".join([r.Identifier, r.Summary, r.ShortDescription]).replace(
            "\n", " "
        )
        texts.append(text)
        identifiers.append(identifier)

        # If the batch size is reached, get embeddings and store them
        if len(texts) == batch_size:
            embeddings.extend(get_embeddings(texts))
            texts.clear()

    # Get embeddings for any remaining texts
    if texts:
        embeddings.extend(get_embeddings(texts))

    # Save embeddings to a JSON Lines file
    with open(EMBEDDINGS, "w") as file:
        for identifier, embedding in zip(identifiers, embeddings):
            embedding_data = {"Identifier": identifier, "Embedding": embedding}
            file.write(json.dumps(embedding_data) + "\n")


class TableSearcher:
    def __init__(self):
        self.embed_df = get_table_embeddings()
        self.df = get_table_list().merge(self.embed_df, on="Identifier")
        self.embeddings = np.vstack(self.df.Embedding.to_list())
        self.index = self._build_faiss_index(self.embeddings)

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def __call__(self, query: str, top_n: int = 3):
        query_embedding = get_embeddings([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_n)
        results = self.df.iloc[indices[0]].copy()
        results["score"] = distances[0]
        return results.to_dict(orient="records")


class TableQuerier:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def clean_response(response: str) -> str:
        # Check if code block markers are present
        if "```" in response:
            # Remove text before the code block
            code_start = response.find("```")
            if code_start > -1:
                response = response[code_start:]

            # Remove markdown code block markers and extract the code block
            code_block = re.search(
                r"```(?:[\w]*\n)?(.*?)```", response, flags=re.DOTALL
            )

            if code_block:
                return code_block.group(1).strip()
            else:
                return ""
        else:
            # If no markdown code block markers, return the response as it is
            return response.strip()

    def translate_query(self, query: str) -> str:
        prompt = f"""df.head(n=3): ```{self.df.head(n=3)}``` query: ```{query}```
give python pandas code to answer the query and assign the result to the 'result' variable.
Note that given is not the complete dataframe just the first 3 rows.
The result of the code must be a pandas dataframe, not a list or numpy array.
Do not prepend or append any comments to the code, because your answer will be evaluated by a python program.
""".replace(
            "\n", " "
        )
        print(prompt)
        messages = [
            {
                "role": "system",
                "content": """You are a highly experienced python pandas programmer.
                Your task it to convert queries to code.
                You respond only with python code.""".replace(
                    "\n", " "
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = get_chat_completion(messages)
        return self.clean_response(response)

    def query(self, query: Union[str, dict]) -> (str, pd.DataFrame):
        pandas_query = self.translate_query(query)

        # Define the restricted environment
        local_vars = {"result": None}
        global_vars = {
            "__builtins__": {
                "getattr": getattr,
                "len": len,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
            },
            "_getattr_": default_guarded_getattr,
            "_getiter_": default_guarded_getiter,
            "_getitem_": default_guarded_getitem,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "pd": pd,
            "df": self.df,
        }

        # Compile and execute the restricted code
        compiled_code = compile_restricted(pandas_query, "<string>", "exec")
        exec(compiled_code, global_vars, local_vars)

        print(pandas_query)
        result: pd.DataFrame = local_vars["result"]

        if isinstance(result, Index):
            result = pd.DataFrame(result)
        elif isinstance(result, Series):
            result = pd.DataFrame(result)
        elif isinstance(result, np.ndarray):
            result = pd.DataFrame(result)
        elif isinstance(result, list):
            result = pd.DataFrame(result)
        elif isinstance(result, int):
            result = pd.DataFrame([result])

        max_rows = 200 // len(result.columns) if len(result.columns) > 0 else 200
        result = result.head(max_rows)

        return pandas_query, result


if __name__ == "__main__":
    ts = TableSearcher()
    print(ts.df.head())
    # results = ts("Personen boven de 18 die nog thuis wonen")
    # print(results)
    df = get_table_data("84669NED")
    print(df.head())
