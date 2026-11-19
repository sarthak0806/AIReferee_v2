"""Open a real browser-like session against a running Streamlit server.

Used to reproduce console errors that only appear in a live session (the test
harness does not start the file watcher or the websocket runtime).

Usage: python -m scripts.streamlit_probe 8610
"""
from __future__ import annotations

import asyncio
import sys

import websockets
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg


async def probe(port: int, seconds: float = 40) -> None:
    async with websockets.connect(f"ws://localhost:{port}/_stcore/stream", max_size=None) as ws:
        msg = BackMsg()
        msg.rerun_script.query_string = ""
        msg.rerun_script.page_script_hash = ""
        await ws.send(msg.SerializeToString())
        kinds: dict[str, int] = {}
        exceptions = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - loop.time()))
                fm = ForwardMsg()
                fm.ParseFromString(raw)
                kind = fm.WhichOneof("type")
                kinds[kind] = kinds.get(kind, 0) + 1
                if kind == "delta" and fm.delta.new_element.WhichOneof("type") == "exception":
                    exceptions.append(fm.delta.new_element.exception.message)
                if kind == "script_finished":
                    break
        except asyncio.TimeoutError:
            print("timed out waiting for script_finished")
        print("messages:", kinds)
        print("exceptions shown in app:", exceptions)


if __name__ == "__main__":
    asyncio.run(probe(int(sys.argv[1]) if len(sys.argv) > 1 else 8610))
