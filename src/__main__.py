"""Entry point: python -m src"""

from src.logging_config import setup_logging

setup_logging()

from src.api import create_app

app = create_app()
app.run(host="127.0.0.1", port=9874, debug=False)
