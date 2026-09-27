import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_handshake_and_tools(self):
        with tempfile.TemporaryDirectory() as root:
            env = {**os.environ, "HYPERTRACE_DB": str(Path(root) / "memory.sqlite3")}
            params = StdioServerParameters(command=sys.executable, args=["-m", "hypertrace_mcp", "serve"], env=env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {tool.name for tool in (await session.list_tools()).tools}
                    self.assertTrue({"browse_memory", "open_memory", "save_feedback", "manage_memory"} <= tools)
                    result = await session.call_tool("browse_memory", {})
                    self.assertFalse(result.isError)
                    payload = result.structuredContent or json.loads(result.content[0].text)
                    self.assertEqual(payload["hypotheses"], [])
                    proposal = {"action": "new", "description": "Run focused tests before completion",
                                "detail": "Run focused tests for changed behavior before reporting completion.",
                                "project_kind": "general coding", "preference_axis": "verification",
                                "likelihood": 3, "baseline": 2}
                    event = {"event_id": "test-follow-up", "excerpt": "Run focused tests for behavior changes",
                             "project": "API coding", "proposals": [proposal]}
                    self.assertTrue((await session.call_tool("save_feedback", event)).isError)
                    proposal["likelihood"] = 5
                    saved = await session.call_tool("save_feedback", event)
                    self.assertFalse(saved.isError)
                    page = await session.call_tool("browse_memory", {"project_kinds": ["general coding"]})
                    cards = page.structuredContent or json.loads(page.content[0].text)
                    self.assertEqual(len(cards["hypotheses"]), 1)
                    repeated = await session.call_tool("save_feedback", event)
                    self.assertFalse(repeated.isError)
                    duplicate = repeated.structuredContent or json.loads(repeated.content[0].text)
                    self.assertTrue(duplicate["duplicate"])


if __name__ == "__main__":
    unittest.main()
