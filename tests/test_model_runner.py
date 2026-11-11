import json
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel

from agents.model_runner import AgentDefinition, execute_agent


class StructuredResponse(BaseModel):
    message: str


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.request: dict[str, Any] | None = None

    async def generate_content(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(text=self.response_text)


class FakeAsyncClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)
        self.aio = SimpleNamespace(models=self.models, aclose=self.aclose)
        self.closed = False

    async def aclose(self):
        self.closed = True


class ModelRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_direct_sdk_call_and_validates_structured_output(self):
        client = FakeAsyncClient('{"message": " verified "}')
        agent = AgentDefinition(
            name="test_agent",
            instruction="Return a structured result.",
            output_key="result",
            output_schema=StructuredResponse,
            model="test-model",
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            with patch("agents.model_runner.genai.Client", return_value=client) as factory:
                result = await execute_agent(agent, {"note": "sample"}, "Check this.")

        factory.assert_called_once_with(api_key="test-key")
        self.assertEqual(result, {"result": {"message": " verified "}})
        self.assertTrue(client.closed)
        request = client.models.request
        if request is None:
            self.fail("Gemini request was not captured.")
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["config"].response_schema, StructuredResponse)
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertIn(json.dumps({"note": "sample"}, indent=2), request["contents"])

    async def test_requires_an_api_key_before_making_a_call(self):
        agent = AgentDefinition(
            name="test_agent",
            instruction="Return text.",
            output_key="result",
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                await execute_agent(agent, {}, "Run.")


if __name__ == "__main__":
    unittest.main()
