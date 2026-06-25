from app.pdf_loader import load_pdf


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, pages):
        self.pages = pages


def test_load_pdf_returns_pages(monkeypatch):
    fake_pages = [
        FakePage("Page one text"),
        FakePage("Page two text"),
    ]

    monkeypatch.setattr(
        "app.pdf_loader.PdfReader",
        lambda path: FakeReader(fake_pages),
    )

    result = load_pdf("fake.pdf")

    assert result == [
        {"page": 1, "text": "Page one text"},
        {"page": 2, "text": "Page two text"},
    ]

def test_load_pdf_handles_none_text(monkeypatch):
    class FakePage:
        def extract_text(self):
            return None

    class FakeReader:
        def __init__(self):
            self.pages = [FakePage()]

    monkeypatch.setattr(
        "app.pdf_loader.PdfReader",
        lambda path: FakeReader(),
    )

    result = load_pdf("fake.pdf")

    assert result == [
        {"page": 1, "text": ""}
    ]