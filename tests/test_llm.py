from app.llm import ask_llm


class FakeResponse:
    class Choice:
        class Message:
            content = (
                "Python is a programming language.\n"
                "Sources:\n"
                "- file1.pdf (page 1)"
            )

        message = Message()

    choices = [Choice()]


def test_ask_llm(monkeypatch):
    def fake_create(model, messages, temperature):
        assert model is not None
        assert temperature == 0.2

        prompt = messages[0]["content"]

        assert "Context:" in prompt
        assert "Python is a language." in prompt
        assert "What is Python?" in prompt

        return FakeResponse()

    monkeypatch.setattr(
        "app.llm.client.chat.completions.create",
        fake_create,
    )

    result = ask_llm(
        context="Python is a language.",
        question="What is Python?",
    )

    assert (
        result
        == "Python is a programming language.\nSources:\n- file1.pdf (page 1)"
    )