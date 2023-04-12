from fastapi.testclient import TestClient
from server.main import BEARER_TOKEN, app
from typing import List

client = TestClient(app)


def test_table_metadata():
    # Replace 'example_table_id' with a valid table_id
    table_id = "83752NED"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = client.post(f"/table_metadata/{table_id}", headers=headers)

    assert response.status_code == 200

    # Check the response structure
    response_data = response.json()
    assert "table_id" in response_data
    assert "column_info" in response_data
    assert "example_data" in response_data

    # Check if table_id matches the requested table_id
    assert response_data["table_id"] == table_id

    # Check the column_info structure
    column_info: List[dict] = response_data["column_info"]
    for column in column_info:
        assert "column_name" in column
        assert "column_type" in column

    # Check the example_data structure
    example_data = response_data["example_data"]
    assert isinstance(example_data, str)
    # Add more checks for example_data if necessary
