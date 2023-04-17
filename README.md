## Installation

Setup virtual environment

    poetry install

Use `.env.example` as a template for `.env` and fill in the required values.

Start server

    poetry run start

This will start the server on `localhost:8123`. Use nginx and e.g. letsencrypt to make it available on the internet
so that the OpenAI plugin runner can access it on https://yourdomain.com/


## CBS opendata

Querying CBS opendata involves two main activities

1. **Table List Retrieval**: Getting the table-list and finding the correct table (dataframe) to query.
   The metadata of a table includes an identifier and a summary. The summary is embedded with ada,
   so we can search for matching keywords in the summary to find the correct table.

2. **Table Record Format**: Getting the record format of the table (dataframe) and the column names. The table is fetched and
   column names as well as summary data are extracted from the table. With this information returned by the
   plugin, the OpenAI plugin runner can hopefully aid the user in answering questions like 'what kind of
   questions can I answer with this table?' or 'what kind of data is in this table?' and formulate a good
   natural language query based on the user query.

3. **Table Querying**: Querying the table (dataframe) with a table identifier and natural language query string. GPT is asked
   to translate the query into one or more pandas operations. The resulting code is then executed in a
   restricted environment, and the resulting data is serialized and returned to the OpenAI plugin runner.
   In case of code-execution errors, GPT is asked two times more to translate the query into pandas
   operations. If it fails multiple times, the error is returned to the OpenAI plugin runner, which hopefully
   transform the error message into something useful for the user.

## Caching

File locations and cache size can be configured in `constants.py`

1. The table list is cached on the local filesystem. To refresh it, delete the `table_list.csv` file.
2. Table summary embeddings are stored on the local filesystem in the `embeddings.jsonl` file, with one embedding per line. To refresh the embeddings, delete the `embeddings.jsonl` file.
3. Table data is cached on the local filesystem in the `cache_directory`, which has a configurable maximum size.

## Speed

Most CBS tables are small enough to be fetched before a timeout occurs.
Queries are executed using pandas (2), which generally performs well and doesn't result in prohibitively
slow operations.

## Tests

To run tests, execute:

    pytest
