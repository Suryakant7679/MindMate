"""The real entry point starts: ``python3 -m backend.server`` (without a model, with a temporary database)."""
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerStartupTests(unittest.TestCase):
    def test_server_main_starts_and_serves_the_api(self):
        with tempfile.TemporaryDirectory() as folder:
            port = free_port()
            process = subprocess.Popen(
                [sys.executable, "-m", "backend.server", "--no-model", "--port", str(port),
                 "--db-path", str(Path(folder) / "chat.sqlite")],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                base = f"http://127.0.0.1:{port}"
                deadline = time.time() + 60
                while True:
                    if process.poll() is not None:
                        self.fail("server exited at startup:\n" + process.stdout.read())
                    try:
                        with urllib.request.urlopen(base + "/api/status", timeout=2) as response:
                            status = json.loads(response.read())
                        break
                    except OSError:
                        if time.time() > deadline:
                            self.fail("server did not answer within 60 s")
                        time.sleep(0.5)
                self.assertTrue(status["persistence"])
                with urllib.request.urlopen(base + "/api/me", timeout=5) as response:
                    self.assertTrue(json.loads(response.read())["consent"]["required"])
                with urllib.request.urlopen(base + "/api/models", timeout=5) as response:
                    self.assertEqual(json.loads(response.read())["models"], [])     # started without models
            finally:
                process.terminate()
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    process.kill()
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()
