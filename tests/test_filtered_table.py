import os

from fastapi.testclient import TestClient

from server.main import BEARER_TOKEN, app

client = TestClient(app)


def test_filtered_table_list_endpoint():
    response = client.post(
        "/filtered_table_list",
        json={"query": "example query"},
        headers={"Authorization": f"Bearer {BEARER_TOKEN}"},
    )

    assert response.status_code == 200
    assert "filtered_tables" in response.json()
    # You can add more assertions based on the expected output of your filter_tables_by_query function
