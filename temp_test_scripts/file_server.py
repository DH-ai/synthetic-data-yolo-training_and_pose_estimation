#!/usr/bin/env python3
import http.server
import socketserver
import os
import urllib.parse
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThreadingHTTPServer(socketserver.ThreadingTCPServer):
    """Handle each request in its own thread so large transfers don't block others"""
    allow_reuse_address = True
    daemon_threads = True


class FileServerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with upload support and directory listing"""

    SERVE_DIR = "./data"  # Change this to your data directory

    def do_GET(self):
        """Handle GET requests (downloads)"""
        # Parse path without query parameters, normalize trailing slash
        path_only = self.path.split('?')[0]
        normalized = path_only.rstrip('/')

        if normalized in ('', '/api/files'):
            logger.info(f"LIST REQUEST from {self.client_address[0]} - path: {path_only}")
            self.list_directory_json()
        else:
            logger.info(f"DOWNLOAD REQUEST from {self.client_address[0]} - file: {path_only}")
            self.serve_file()

    def do_POST(self):
        """Handle POST requests (uploads)"""
        if self.path == '/upload':
            logger.info(f"UPLOAD REQUEST from {self.client_address[0]}")
            self.handle_upload()
        else:
            logger.warning(f"INVALID POST REQUEST from {self.client_address[0]} - path: {self.path}")
            self.send_error(404, "Not Found")

    def serve_file(self):
        """Serve a file for download"""
        file_path = urllib.parse.unquote(self.path.lstrip('/'))
        full_path = os.path.join(self.SERVE_DIR, file_path)

        # Security: prevent path traversal
        try:
            full_path = os.path.abspath(full_path)
            serve_dir = os.path.abspath(self.SERVE_DIR)
            if not full_path.startswith(serve_dir):
                logger.warning(f"PATH TRAVERSAL ATTEMPT from {self.client_address[0]} - path: {file_path}")
                self.send_error(403, "Forbidden")
                return
        except:
            self.send_error(400, "Bad Request")
            return

        if not os.path.exists(full_path):
            logger.warning(f"FILE NOT FOUND from {self.client_address[0]} - file: {file_path}")
            self.send_error(404, "File Not Found")
            return

        if os.path.isdir(full_path):
            logger.info(f"DIRECTORY LIST from {self.client_address[0]} - dir: {file_path}")
            self.list_directory_json()
            return

        try:
            file_size = os.path.getsize(full_path)
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Length', str(file_size))
            self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(full_path)}"')
            self.end_headers()

            # Stream in 1MB chunks so large files (10GB+) never load fully into memory
            chunk_size = 1024 * 1024
            with open(full_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

            size_mb = file_size / (1024*1024)
            logger.info(f"DOWNLOAD COMPLETE from {self.client_address[0]} - file: {file_path} ({size_mb:.2f} MB)")
        except (BrokenPipeError, ConnectionResetError):
            logger.warning(f"DOWNLOAD ABORTED by client {self.client_address[0]} - file: {file_path}")
        except Exception as e:
            logger.error(f"DOWNLOAD ERROR from {self.client_address[0]} - file: {file_path} - error: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")

    def handle_upload(self):
        """Handle file uploads"""
        content_length = int(self.headers.get('Content-Length', 0))

        if content_length == 0:
            logger.warning(f"EMPTY UPLOAD from {self.client_address[0]}")
            self.send_error(400, "No file provided")
            return

        filename = self.headers.get('X-Filename', 'upload')
        filepath = os.path.join(self.SERVE_DIR, filename)

        size_mb = content_length / (1024*1024)
        logger.info(f"UPLOAD STARTING from {self.client_address[0]} - file: {filename} ({size_mb:.2f} MB)")

        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

            bytes_read = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            with open(filepath, 'wb') as f:
                while bytes_read < content_length:
                    chunk = self.rfile.read(min(chunk_size, content_length - bytes_read))
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_read += len(chunk)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                'status': 'success',
                'filename': filename,
                'size': bytes_read
            })
            self.wfile.write(response.encode())

            uploaded_mb = bytes_read / (1024*1024)
            logger.info(f"UPLOAD COMPLETE from {self.client_address[0]} - file: {filename} ({uploaded_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"UPLOAD ERROR from {self.client_address[0]} - file: {filename} - error: {str(e)}")
            self.send_error(500, f"Upload failed: {str(e)}")

    def list_directory_json(self):
        """List directory contents as JSON"""
        try:
            path = urllib.parse.unquote(self.path.lstrip('/'))
            if path.startswith('api/files'):
                path = ''
            full_path = os.path.join(self.SERVE_DIR, path)

            full_path = os.path.abspath(full_path)
            serve_dir = os.path.abspath(self.SERVE_DIR)
            if not full_path.startswith(serve_dir):
                logger.warning(f"PATH TRAVERSAL in LIST from {self.client_address[0]} - path: {path}")
                self.send_error(403, "Forbidden")
                return

            if not os.path.isdir(full_path):
                logger.warning(f"INVALID PATH in LIST from {self.client_address[0]} - path: {path}")
                self.send_error(404, "Not Found")
                return

            items = []
            for item in sorted(os.listdir(full_path)):
                item_path = os.path.join(full_path, item)
                rel_path = os.path.relpath(item_path, self.SERVE_DIR)

                if os.path.isdir(item_path):
                    items.append({
                        'name': item,
                        'type': 'directory',
                        'path': rel_path
                    })
                else:
                    size = os.path.getsize(item_path)
                    items.append({
                        'name': item,
                        'type': 'file',
                        'path': rel_path,
                        'size': size,
                        'size_mb': round(size / (1024*1024), 2)
                    })

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(items, indent=2).encode())
            logger.debug(f"Listed {len(items)} items from {self.client_address[0]}")
        except Exception as e:
            logger.error(f"LIST ERROR from {self.client_address[0]} - error: {str(e)}")
            self.send_error(500, f"Error: {str(e)}")

    def log_message(self, format, *args):
        """Suppress default logging since we use our own"""
        pass


def main():
    parser = argparse.ArgumentParser(description="Local network file server")
    parser.add_argument('-d', '--dir', default='./data',
                        help='Directory to serve files from (default: ./data)')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port to listen on (default: 8000)')
    args = parser.parse_args()

    FileServerHandler.SERVE_DIR = args.dir
    PORT = args.port

    # Create the served directory if it doesn't exist
    os.makedirs(FileServerHandler.SERVE_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("File Server Starting")
    logger.info("=" * 60)
    logger.info(f"Serving files from: {os.path.abspath(FileServerHandler.SERVE_DIR)}")
    logger.info(f"Port: {PORT}")

    with ThreadingHTTPServer(("", PORT), FileServerHandler) as httpd:
        logger.info(f"Server listening on http://0.0.0.0:{PORT}")

        # Try to show local network IP (UDP-connect trick avoids returning 127.0.0.1)
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            logger.info(f"Access from another computer: http://{local_ip}:{PORT}")
        except Exception:
            logger.warning("Could not determine local IP address")

        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    main()
