import pytest
from fastapi.testclient import TestClient

from server.main import BEARER_TOKEN, app

client = TestClient(app)


def test_query_table_data():
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = client.post(
        "/query_table_data",
        json={
            "table_id": "70947ned",
            "natural_language_query": "Welke kolommen heeft deze tabel?",
        },
        headers=headers,
    )

    processed_query = response.json()["processed_query"]
    print(processed_query)
    csv_data = response.json()["data"]


def test_query_table_data_error_80190eng():
    response = client.post(
        "/query_table_data",
        json={"table_id": "80190eng", "natural_language_query": ""},
        headers={"Authorization": f"Bearer {BEARER_TOKEN}"},
    )

    assert response.status_code == 500
    assert "detail" in response.json()
    assert "Downloading table '80190eng' failed" in response.json()["detail"]
    assert "404 Client Error" in response.json()["detail"]
