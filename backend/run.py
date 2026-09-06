"""
Single-instruction launch of the whole system (backend API + frontend UI
served from the same FastAPI process, same port):

    python run.py

This opens http://localhost:8000 in your default browser automatically
once the server is confirmed to be accepting connections. Pass --no-browser
to skip that (e.g. running headless on a remote machine).
"""
import sys
import time
import threading
import socket
import webbrowser
import uvicorn

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def _wait_for_server_then_open(host: str, port: int, url: str, timeout_s: float = 15.0):
    """Poll the port until uvicorn is actually accepting connections, then
    open the browser -- avoids the 'connection refused' flash you get from
    opening a browser tab before the server has finished starting up."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.2)


if __name__ == "__main__":
    open_browser = "--no-browser" not in sys.argv

    if open_browser:
        threading.Thread(
            target=_wait_for_server_then_open,
            args=(HOST, PORT, URL),
            daemon=True,
        ).start()

    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)
