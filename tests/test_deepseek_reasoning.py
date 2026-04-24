"""Tests for DeepSeek thinking-mode tool-call compatibility."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDeepSeekReasoningContent(unittest.TestCase):
    def test_stream_reasoning_content_is_preserved(self):
        from llmcore import _parse_openai_sse

        lines = iter([
            b'data: {"choices":[{"delta":{"reasoning_content":"Need a tool. "}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"code_run","arguments":"{\\\"type\\\":\\\"bash\\\"}"}}]}}]}',
            b'data: [DONE]',
        ])

        gen = _parse_openai_sse(lines)
        with self.assertRaises(StopIteration) as stop:
            while True:
                next(gen)

        blocks = stop.exception.value
        self.assertEqual(blocks[0], {"type": "thinking", "thinking": "Need a tool. "})
        self.assertEqual(blocks[1]["type"], "tool_use")
        self.assertEqual(blocks[1]["id"], "call_1")

    def test_assistant_thinking_block_round_trips_to_openai_reasoning_content(self):
        from llmcore import _msgs_claude2oai

        messages = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Need a tool."},
                {"type": "text", "text": "Let me inspect it."},
                {"type": "tool_use", "id": "call_1", "name": "code_run", "input": {"type": "bash", "script": "pwd"}},
            ],
        }]

        converted = _msgs_claude2oai(messages)

        self.assertEqual(converted[0]["reasoning_content"], "Need a tool.")
        self.assertEqual(converted[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(json.loads(converted[0]["tool_calls"][0]["function"]["arguments"])["script"], "pwd")


if __name__ == "__main__":
    unittest.main()
