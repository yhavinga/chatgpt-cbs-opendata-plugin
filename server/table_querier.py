import re

import numpy as np
from pandas import Index, Series
from RestrictedPython import compile_restricted
from RestrictedPython.Eval import (default_guarded_getattr,
                                   default_guarded_getitem,
                                   default_guarded_getiter)
from RestrictedPython.Guards import (full_write_guard,
                                     guarded_iter_unpack_sequence,
                                     guarded_unpack_sequence)

from server.logger import logger
from server.pandas_custom_display import pd
from server.table import table_summary
from services.openai import get_chat_completion


class CBSTableQuerier:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def get_code_from_completion(completion: str) -> str:
        # Check if code block markers are present
        if "```" in completion:
            # Remove text before the code block
            code_start = completion.find("```")
            if code_start > -1:
                completion = completion[code_start:]

            # Remove markdown code block markers and extract the code block
            code_block = re.search(
                r"```(?:[\w]*\n)?(.*?)```", completion, flags=re.DOTALL
            )

            if code_block:
                return code_block.group(1).strip()
            else:
                return ""
        else:
            return completion.strip()

    def translate_query(
        self, query: str, previous_pandas: str = None, previous_error: str = None
    ) -> str:
        df_head = re.sub(r"\s+", " ", str(self.df.head(n=3)))
        #         prompt = f"""df.head(n=3): ```{df_head}``` query: ```{query}```
        #         NB! the data above is an example so you know the format. Do not come to the conclusion that the query cannot be answered
        # because the data is incomplete. Your answer will run on the complete dataframe.
        # Your task it to write one or more python pandas instructions on this dataframe to answer the user query.
        # Think step by step how to massage the dataframe to match the user query.
        # Return one or more python code lines with pandas dataframe operations to answer the query and assign the result to the 'result' variable.
        # The result of the code must be a pandas dataframe, not a list or numpy array.
        # """.replace(
        #             "\n", " "
        #         )
        df_summary = re.sub(r"\s+", " ", str(table_summary(self.df)))
        previous_stuff = (
            f"""Your previous query was: ```{previous_pandas}``` and it failed with the error: ```{previous_error}```."""
            if previous_pandas and previous_error
            else ""
        )
        prompt = f"""Given a df with column statistics and top 5 most common values ```{df_summary}```
and df.head(n=3): ```{df_head}```
Your task it to write one or more python pandas instructions on this dataframe to answer this user query
```{query}```
{previous_stuff}
Think step by step how to massage the dataframe to match the user query.
Beware of constant values for categorical columns. Do all string comparisons case insensitive.
Also try to best guess which existing categories the user is referring to. For instance if the user says 'kippen' and the table has 'pluimvee'.
Do not use the 'print' functions in your code.
Return one or more python code lines with pandas dataframe operations to answer the query and assign the result to the 'result' variable.
If there are multiple questions, do not return multiple code blocks. Instead, answer all questions in a single code block.
It is imperative that the last line of your code should assign a value to the 'result' variable.
Even if you compute two intermediate values, then the last line should make sure they are assigned to the 'result' variable, preferably with a meaningful name and DataFrame type.
""".replace(
            "\n", " "
        )

        print(prompt)
        messages = [
            {
                "role": "system",
                "content": """Answer as a highly experienced python pandas programmer without attitude.
Your task is to convert natural language queries to python pandas code.
Assume the user provides example rows and the code in your response will run on the complete data.
Make the best possible assumption mapping the query to the columns.""".replace(
                    "\n", " "
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = get_chat_completion(messages)
        return self.get_code_from_completion(response)

    def query(self, query: str) -> (str, pd.DataFrame):
        max_retries = 3
        retries = 0
        pandas_query = None
        previous_pandas = None
        previous_error = None
        local_vars = {"result": None}

        while retries < max_retries and local_vars["result"] is None:
            retries += 1
            pandas_query = self.translate_query(query, previous_pandas, previous_error)
            # Define the restricted environment
            global_vars = {
                "__builtins__": {
                    "getattr": getattr,
                    "len": len,
                    "range": range,
                    "int": int,
                    "float": float,
                    "str": str,
                    "write": lambda x: x,
                },
                "_getattr_": default_guarded_getattr,
                "_getiter_": default_guarded_getiter,
                "_getitem_": default_guarded_getitem,
                "_write_": full_write_guard,
                "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                "_unpack_sequence_": guarded_unpack_sequence,
                "pd": pd,
                "df": self.df,
            }

            # Compile and execute the restricted code
            compiled_code = compile_restricted(pandas_query, "<string>", "exec")
            try:
                logger.info(f"Executing query: {pandas_query}")
                exec(compiled_code, global_vars, local_vars)
                break
            except Exception as e:
                if retries < max_retries:
                    logger.info(f"Error n. {retries}: {str(e)} -- Retrying...")
                    previous_pandas = pandas_query
                    previous_error = str(e)
                else:
                    logger.error(f"Error: {str(e)}")
                    return pandas_query, pd.DataFrame()

        result = local_vars["result"]

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
        elif isinstance(result, np.float64):
            result = pd.DataFrame([result])
        elif isinstance(result, np.int64):
            result = pd.DataFrame([result])
        elif isinstance(result, float):
            result = pd.DataFrame([result])
        elif isinstance(result, str):
            result = pd.DataFrame([result])
        elif isinstance(result, bool):
            result = pd.DataFrame([result])
        elif isinstance(result, dict):
            result = pd.DataFrame([result])
        elif isinstance(result, tuple):
            result = pd.DataFrame([list(result)])

        max_rows = 200 // len(result.columns) if len(result.columns) > 0 else 200
        result = result.head(max_rows)

        return pandas_query, result
