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


if __name__ == "__main__":
    unittest.main()
