"""PyInstaller entry point — starts Flask server and opens browser."""

import multiprocessing
import socket
import sys
import threading
import time
import traceback
import webbrowser

HOST = "127.0.0.1"
PORT = 9876
URL = f"http://{HOST}:{PORT}"


def is_server_running() -> bool:
    """Check if something is already listening on our port."""
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def open_browser():
    """Wait briefly for the server to start, then open the browser."""
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # If server is already running, just open the browser and exit
    if is_server_running():
        print(f"GafferAI is already running at {URL}")
        webbrowser.open(URL)
        sys.exit(0)

    try:
        print(f"Starting GafferAI at {URL}")
        print("Close this window to stop the server.\n")

        threading.Thread(target=open_browser, daemon=True).start()

        from src.api import create_app

        app = create_app()
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except Exception:
        traceback.print_exc()
        input("\nPress Enter to exit...")
