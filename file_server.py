#!/usr/bin/env python3
import http.server
import socketserver
import os
import urllib.parse
import json
from pathlib import Path

class FileServerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with upload support and directory listing"""

    SERVE_DIR = "./data"  # Change this to your data directory

    def do_GET(self):
        """Handle GET requests (downloads)"""
        if self.path == '/':
            self.list_directory_json()
        elif self.path == '/api/files':
            self.list_directory_json()
        else:
            self.serve_file()

    def do_POST(self):
        """Handle POST requests (uploads)"""
        if self.path == '/upload':
            self.handle_upload()
        else:
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
                self.send_error(403, "Forbidden")
                return
        except:
            self.send_error(400, "Bad Request")
            return

        if not os.path.exists(full_path):
            self.send_error(404, "File Not Found")
            return

        if os.path.isdir(full_path):
            self.list_directory_json()
            return

        try:
            file_size = os.path.getsize(full_path)
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Length', str(file_size))
            self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(full_path)}"')
            self.end_headers()

            with open(full_path, 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, f"Error: {str(e)}")

    def handle_upload(self):
        """Handle file uploads"""
        content_length = int(self.headers.get('Content-Length', 0))

        if content_length == 0:
            self.send_error(400, "No file provided")
            return

        filename = self.headers.get('X-Filename', 'upload')
        filepath = os.path.join(self.SERVE_DIR, filename)

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
        except Exception as e:
            self.send_error(500, f"Upload failed: {str(e)}")

    def list_directory_json(self):
        """List directory contents as JSON"""
        try:
            path = urllib.parse.unquote(self.path.lstrip('/'))
            full_path = os.path.join(self.SERVE_DIR, path)

            full_path = os.path.abspath(full_path)
            serve_dir = os.path.abspath(self.SERVE_DIR)
            if not full_path.startswith(serve_dir):
                self.send_error(403, "Forbidden")
                return

            if not os.path.isdir(full_path):
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
        except Exception as e:
            self.send_error(500, f"Error: {str(e)}")

    def log_message(self, format, *args):
        """Log requests"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    # Create data directory if it doesn't exist
    os.makedirs(FileServerHandler.SERVE_DIR, exist_ok=True)

    PORT = 8000

    with socketserver.TCPServer(("", PORT), FileServerHandler) as httpd:
        print(f"File Server running on http://0.0.0.0:{PORT}")
        print(f"Serving files from: {os.path.abspath(FileServerHandler.SERVE_DIR)}")
        print(f"\nAccess from another computer:")

        # Try to show local network IP
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"  http://{local_ip}:{PORT}")
        except:
            pass

        print(f"\nUsage:")
        print(f"  Download: GET http://server:{PORT}/filename")
        print(f"  List files: GET http://server:{PORT}/api/files")
        print(f"  Upload: POST http://server:{PORT}/upload with file in body")
        print(f"\nPress Ctrl+C to stop\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
