from typing import List, Optional

from pydantic import BaseModel


class TableInfo(BaseModel):
    Identifier: str
    Title: str
    ShortTitle: str
    ShortDescription: str
    Period: str
    Updated: Optional[str] = None
    Language: Optional[str] = None
    Catalog: Optional[str] = None
    Frequency: Optional[str] = None
    Summary: Optional[str] = None
    Modified: Optional[str] = None
    MetaDataModified: Optional[str] = None
    OutputStatus: Optional[str] = None
    Source: Optional[str] = None
    ApiUrl: Optional[str] = None
    FeedUrl: Optional[str] = None
    DefaultPresentation: Optional[str] = None
    DefaultSelection: Optional[str] = None
    GraphTypes: Optional[str] = None
    RecordCount: Optional[int] = None
    ColumnCount: Optional[int] = None
    SearchPriority: Optional[str] = None


class TableInfoWithScore(TableInfo):
    score: float


class FilteredTableListRequest(BaseModel):
    query: str


class FilteredTableListResponse(BaseModel):
    filtered_tables: List[TableInfoWithScore]
