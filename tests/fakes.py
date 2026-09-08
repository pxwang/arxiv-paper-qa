"""Test doubles shared across the test suite."""

from langchain_core.runnables import Runnable


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM(Runnable):
    """Stands in for ChatOllama. Subclasses Runnable so `PROMPT | FakeLLM()`
    produces a real LangChain RunnableSequence with correct invoke semantics,
    rather than relying on how a bare Mock happens to get coerced."""

    def __init__(self, response_text: str = "Fake answer."):
        self.response_text = response_text
        self.invocations = []

    def invoke(self, input, config=None, **kwargs):
        self.invocations.append(input)
        return FakeMessage(self.response_text)
