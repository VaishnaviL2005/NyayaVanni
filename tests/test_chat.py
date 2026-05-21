import pytest

def test_general_chat_streaming(test_client):
    """Test the /api/chat/general endpoint returns a streaming response with content."""
    response = test_client.post("/api/chat/general", json={"message": "Hello"})
    # The response should be a StreamingResponse (status code 200)
    assert response.status_code == 200
    # Read the streamed content
    content = b"".join([chunk for chunk in response.iter_content()])
    # Ensure some data was returned (non-empty)
    assert content, "Streaming response returned empty content"
