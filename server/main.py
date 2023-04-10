import os
from io import StringIO

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from models.api import (FilteredTableListRequest, FilteredTableListResponse,
                        TableColumnInfo, TableDataQueryRequest,
                        TableDataQueryResponse, TableInfo,
                        TableMetadataResponse)
from server.table import TableQuerier, TableSearcher, get_table_data

load_dotenv(".env")

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


class CustomHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


async def custom_http_exception_handler(request: Request, exc: CustomHTTPException):
    print(exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app = FastAPI(dependencies=[Depends(validate_token)])
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema,
# found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="CBS Opendata Plugin API",
    description="An API for querying CBS Opendata based on natural language queries",
    version="1.0.0",
    servers=[{"url": "https://chatdata.nl"}],
    dependencies=[Depends(validate_token)],
)
app.mount("/sub", sub_app)
app.add_exception_handler(CustomHTTPException, custom_http_exception_handler)

table_searcher = TableSearcher()


async def filter_tables_by_query(query: str):
    filtered_table_list = table_searcher(query)
    return filtered_table_list


@app.post(
    "/filtered_table_list",
    response_model=FilteredTableListResponse,
)
async def filtered_table_list(
    request: FilteredTableListRequest = Body(...),
):
    try:
        filtered_tables = await filter_tables_by_query(
            request.query,
        )
        return FilteredTableListResponse(filtered_tables=filtered_tables)

    except Exception as e:
        error_message = str(e)
        raise CustomHTTPException(status_code=500, detail=error_message)


@sub_app.post(
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
async def filtered_table_list(
    request: FilteredTableListRequest = Body(...),
):
    try:
        filtered_tables = await filter_tables_by_query(
            request.query,
        )
        return FilteredTableListResponse(filtered_tables=filtered_tables)
    except Exception as e:
        error_message = str(e)
        raise CustomHTTPException(status_code=500, detail=error_message)


@app.post(
    "/table_metadata/{table_id}",
    response_model=TableMetadataResponse,
)
async def table_metadata(table_id: str):
    try:
        df = get_table_data(table_id)
        column_info = [
            TableColumnInfo(column_name=col, column_type=str(df[col].dtype))
            for col in df.columns
        ]
        csv_buffer = StringIO()
        df.head(5).to_csv(csv_buffer, index=False)
        example_data = csv_buffer.getvalue()

        return TableMetadataResponse(
            table_id=table_id, column_info=column_info, example_data=example_data
        )
    except Exception as e:
        error_message = str(e)
        raise CustomHTTPException(status_code=510, detail=error_message)


@sub_app.post(
    "/table_metadata/{table_id}",
    response_model=TableMetadataResponse,
    description="""Returns metadata for a table given an identifier.
Metadata is the column information and example data.
""".replace(
        "\n", " "
    ),
)
async def table_metadata(table_id: str):
    try:
        df = get_table_data(table_id)
        column_info = [
            TableColumnInfo(column_name=col, column_type=str(df[col].dtype))
            for col in df.columns
        ]
        csv_buffer = StringIO()
        df.head(5).to_csv(csv_buffer, index=False)
        example_data = csv_buffer.getvalue()

        return TableMetadataResponse(
            table_id=table_id, column_info=column_info, example_data=example_data
        )
    except Exception as e:
        error_message = str(e)
        raise CustomHTTPException(status_code=510, detail=error_message)


@app.post(
    "/query_table_data",
    response_model=TableDataQueryResponse,
)
async def query_table_data(request: TableDataQueryRequest):
    try:
        df = get_table_data(request.table_id)

        # Initialize the TableQuerier with the DataFrame
        table_querier = TableQuerier(df)

        # Perform the natural language query
        pandas_query, result_df = table_querier.query(request.natural_language_query)

        # Serialize the resulting data as CSV
        csv_buffer = StringIO()
        result_df.to_csv(csv_buffer, index=False)
        data = csv_buffer.getvalue()
        return TableDataQueryResponse(processed_query=pandas_query, data=data)
    except Exception as e:
        error_message = str(e)
        raise CustomHTTPException(status_code=510, detail=error_message)


@sub_app.post(
    "/query_table_data",
    response_model=TableDataQueryResponse,
    description="""Accepts a CBS opendata table identifier and a natural language query on this table.
Table identifiers and dataset time periods can be found with /filtered_table_list endpoint.
Split queries if ResponseTooLargeError occurs.""".replace(
        "\n", " "
    ),
)
async def query_table_data(request: TableDataQueryRequest):
    try:
        df = get_table_data(request.table_id)

        # Initialize the TableQuerier with the DataFrame
        table_querier = TableQuerier(df)

        # Perform the natural language query
        pandas_query, result_df = table_querier.query(request.natural_language_query)

        # Serialize the resulting data as CSV
        csv_buffer = StringIO()
        result_df.to_csv(csv_buffer, index=False)
        data = csv_buffer.getvalue()
        return TableDataQueryResponse(processed_query=pandas_query, data=data)
    except Exception as e:
        error_message = str(e)
        raise CustomHTTPException(status_code=510, detail=error_message)


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8123, reload=True)
