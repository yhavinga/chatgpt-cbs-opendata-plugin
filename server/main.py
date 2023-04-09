import os
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from models.api import FilteredTableListRequest, FilteredTableListResponse, TableInfo
from server.table import TableSearcher

load_dotenv(".env")

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


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
    # try:
    filtered_tables = await filter_tables_by_query(
        request.query,
    )
    return FilteredTableListResponse(filtered_tables=filtered_tables)
    # except Exception as e:
    #
    #     print("Error:", e)
    #     raise HTTPException(status_code=500, detail="Internal Service Error")


@sub_app.post(
    "/filtered_table_list",
    response_model=FilteredTableListResponse,
    # NOTE: We are describing the shape of the API endpoint input due to a current limitation in parsing arrays of objects from OpenAPI schemas. This will not be necessary in the future.
    description="Accepts search strings array each with a natural language query for a table. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
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
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8123, reload=True)
