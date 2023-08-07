import os
import urllib.parse
import re
from pathlib import Path
from flask import Flask, send_from_directory

from toolboxv2 import get_logger

app = Flask(__name__)

pattern = re.compile('.png|.jpg|.jpeg|.js|.css|.ico|.gif|.svg|.wasm', re.IGNORECASE)


def serve_app_change_dir():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


logger = get_logger()


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_app_func(path):
    try:
        request_file_path = Path(path)
        ext = request_file_path.suffix
        if not request_file_path.is_file() and not pattern.match(ext):
            path = 'test.html'

        logger.info(f"Sending : {path} ")
        serve_app_func
        return send_from_directory('./', path)
    except Exception as e:
        logger.error("Error processing request: %s", e)


if __name__ == '__main__':
    app.run(port=8080)