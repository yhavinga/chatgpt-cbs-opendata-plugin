import re
from io import StringIO

import uvicorn
from fastapi import APIRouter, Body

from models.api import (FilteredTableListRequest, FilteredTableListResponse,
                        TableColumnInfo, TableDataQueryRequest,
                        TableDataQueryResponse, TableInfoWithScore,
                        TableMetadataResponse)
from server.table import get_table_data, table_summary
from server.table_list import CBSTableEmbedder, CBSTableSearcher
from server.table_querier import CBSTableQuerier

router = APIRouter()
sub_router = APIRouter()
table_searcher = CBSTableSearcher(CBSTableEmbedder())


async def handle_filtered_table_list(request: FilteredTableListRequest):
    filtered_tables = [
        TableInfoWithScore(**table) for table in table_searcher(request.query)
    ]
    return FilteredTableListResponse(filtered_tables=filtered_tables)


@router.post("/filtered_table_list", response_model=FilteredTableListResponse)
@sub_router.post(
    "/filtered_table_list",
    response_model=FilteredTableListResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="""Accepts a natural language query to find matching CBS opendata tables.
Returns the table identifiers and summaries including time periods.
Break down complex questions into sub-questions.
Split queries if ResponseTooLargeError occurs.""".replace(
        "\n", " "
    ),
)
async def filtered_table_list(request: FilteredTableListRequest = Body(...)):
    return await handle_filtered_table_list(request)


async def handle_table_metadata(table_id: str) -> TableMetadataResponse:
    df = get_table_data(table_id)
    column_info = [
        TableColumnInfo(column_name=col, column_type=str(df[col].dtype))
        for col in df.columns
    ]
    csv_buffer = StringIO()
    # df.head(1).to_csv(csv_buffer, float_format="%.1f", na_rep="NA", index=False)
    table_summary(df).to_csv(csv_buffer, float_format="%.1f", na_rep="NA")
    summary_data = csv_buffer.getvalue()
    summary_data = re.sub(r"\s+", " ", summary_data)

    return TableMetadataResponse(
        table_id=table_id, column_info=column_info, example_data=summary_data
    )


@router.post(
    "/table_metadata/{table_id}",
    response_model=TableMetadataResponse,
)
@sub_router.post(
    "/table_metadata/{table_id}",
    response_model=TableMetadataResponse,
    description="""Returns metadata for a table given an identifier.
Metadata is the column information and example data.
""".replace(
        "\n", " "
    ),
)
async def table_metadata(table_id: str):
    return await handle_table_metadata(table_id)


async def handle_query_table_data(request: TableDataQueryRequest):
    df = get_table_data(request.table_id)
    table_querier = CBSTableQuerier(df)
    pandas_query, result_df = table_querier.query(request.natural_language_query)
    csv_buffer = StringIO()
    result_df.to_csv(csv_buffer, index=False)
    data = csv_buffer.getvalue()
    return TableDataQueryResponse(processed_query=pandas_query, data=data)


@router.post(
    "/query_table_data",
    response_model=TableDataQueryResponse,
)
@sub_router.post(
    "/query_table_data",
    response_model=TableDataQueryResponse,
    description="""Accepts a CBS opendata table identifier and a natural language query on this table.
Table identifiers and dataset time periods can be found with /filtered_table_list endpoint.
Break down complex questions into sub-questions.
Split queries if ResponseTooLargeError occurs.""".replace(
        "\n", " "
    ),
)
async def query_table_data(request: TableDataQueryRequest):
    return await handle_query_table_data(request)


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8123, reload=True)
