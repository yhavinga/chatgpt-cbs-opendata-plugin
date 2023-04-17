import os
import traceback

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from server.logger import log_requests, logger
from server.routes import router, sub_router

load_dotenv()

bearer_scheme = HTTPBearer()
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
assert BEARER_TOKEN is not None


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials


class AppExceptionHandlerMiddleware:
    async def __call__(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=510, content={"detail": str(e)})


app = FastAPI(dependencies=[Depends(validate_token)])
app.middleware("http")(log_requests)
app.middleware("http")(AppExceptionHandlerMiddleware())
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")
app.include_router(router)

# Create a sub-application, in order to access just the query endpoint in an OpenAPI schema,
# found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="CBS Opendata Plugin API",
    description="An API for querying CBS Opendata based on natural language queries",
    version="1.0.0",
    servers=[{"url": "https://chatdata.nl"}],
    dependencies=[Depends(validate_token)],
)
sub_app.include_router(sub_router)
app.mount("/sub", sub_app)


def start():
    uvicorn.run("server.main:app", host="127.0.0.1", port=8123, reload=True)
