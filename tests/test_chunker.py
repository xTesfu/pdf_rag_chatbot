from app.chunker import chunk_text


def test_chunk_text_single_page_no_split():
    pages = [
        {"text": "This is a short text.", "page": 1}
    ]

    result = chunk_text(pages, document_name="doc1")

    assert len(result) == 1
    assert result[0]["text"] == "This is a short text."
    assert result[0]["page"] == 1
    assert result[0]["document"] == "doc1"


def test_chunk_text_multiple_pages():
    pages = [
        {"text": "Page one content.", "page": 1},
        {"text": "Page two content.", "page": 2}
    ]

    result = chunk_text(pages, document_name="doc2")

    pages_returned = [r["page"] for r in result]
    assert 1 in pages_returned
    assert 2 in pages_returned
    assert all(r["document"] == "doc2" for r in result)


def test_chunk_text_empty_input():
    pages = []

    result = chunk_text(pages)

    assert result == []


def test_chunk_text_structure_keys():
    pages = [
        {"text": "Some sample text for testing chunking behavior.", "page": 5}
    ]

    result = chunk_text(pages, document_name="mydoc")

    assert isinstance(result, list)
    assert "text" in result[0]
    assert "page" in result[0]
    assert "document" in result[0]


def test_chunk_text_chunking_happens():
    # Force long text to ensure splitting happens
    long_text = "word " * 1000  # large enough to trigger chunking

    pages = [
        {"text": long_text, "page": 1}
    ]

    result = chunk_text(pages, document_name="doc")

    # Should produce multiple chunks
    assert len(result) > 1

    # All chunks should preserve metadata
    for chunk in result:
        assert chunk["page"] == 1
        assert chunk["document"] == "doc"
        assert isinstance(chunk["text"], str)