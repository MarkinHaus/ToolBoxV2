import socketserver
#!/usr/bin/env python
import os
# Inspired by https://gist.github.com/jtangelder/e445e9a7f5e31c220be6
# Python3 http.server for Single Page Application

import urllib.parse
import http.server
import re
from pathlib import Path


pattern = re.compile('.png|.jpg|.jpeg|.js|.css|.ico|.gif|.svg|.wasm', re.IGNORECASE)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class AppServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        request_file_path = Path(url_parts.path.strip("/"))

        ext = request_file_path.suffix
        if not request_file_path.is_file() and not pattern.match(ext):
            self.path = '/index.html'
        self.path = 'app'+self.path
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

if __name__ == '__main__':
    httpd = socketserver.TCPServer(('0.0.0.0', 5000), AppServerHandler)
    print("Starting")
    httpd.serve_forever()
    print("Don")


