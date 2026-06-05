#!/usr/bin/env python3
"""
Simple file transfer client for the local file server
Usage:
  python file_client.py download <server:port> <filename>
  python file_client.py upload <server:port> <filepath>
  python file_client.py list <server:port>
"""

import sys
import os
import requests
from pathlib import Path


def download_file(server, filename, output=None):
    """Download a file from server"""
    if not output:
        output = filename.split('/')[-1]

    url = f"http://{server}/{filename}"
    print(f"Downloading {filename} from {server}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"  Progress: {pct:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

        print(f"\nDownloaded to: {output}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def upload_file(server, filepath):
    """Upload a file to server"""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath)

    print(f"Uploading {filename} ({file_size / (1024*1024):.2f} MB) to {server}...")

    try:
        with open(filepath, 'rb') as f:
            headers = {'X-Filename': filename}
            response = requests.post(
                f"http://{server}/upload",
                data=f,
                headers=headers,
                stream=True
            )
            response.raise_for_status()

        print(f"Upload successful: {response.json()}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def list_files(server):
    """List files on server"""
    try:
        response = requests.get(f"http://{server}/api/files")
        response.raise_for_status()
        files = response.json()

        print(f"\nFiles on {server}:")
        print("-" * 60)
        for item in files:
            if item['type'] == 'directory':
                print(f"  [DIR]  {item['name']}/")
            else:
                size_mb = item.get('size_mb', item['size'] / (1024*1024))
                print(f"  {item['name']:40} {size_mb:10.2f} MB")
        print("-" * 60)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    server = sys.argv[2]

    if command == 'download':
        if len(sys.argv) < 4:
            print("Usage: python file_client.py download <server:port> <filename> [output]")
            sys.exit(1)
        filename = sys.argv[3]
        output = sys.argv[4] if len(sys.argv) > 4 else None
        download_file(server, filename, output)
    elif command == 'upload':
        if len(sys.argv) < 4:
            print("Usage: python file_client.py upload <server:port> <filepath>")
            sys.exit(1)
        filepath = sys.argv[3]
        upload_file(server, filepath)
    elif command == 'list':
        list_files(server)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
