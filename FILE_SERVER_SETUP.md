# Local Network File Server

A simple Python-based file server for sharing large files (10GB+) between computers on a local network.

## Quick Start

### 1. Start the Server

Point it at the directory you want to share (no need to edit the file):

```bash
# Serve a specific directory on a specific port
python file_server.py --dir /path/to/your/data --port 8000

# Defaults: --dir ./data  --port 8000
python file_server.py
```

The server will start on `http://0.0.0.0:8000` and show your local network IP.

**Output example:**
```
============================================================
File Server Starting
============================================================
Serving files from: /home/dhruv/.../image_data_win
Port: 8000
Server listening on http://0.0.0.0:8000
Access from another computer: http://192.168.137.15:8000
============================================================
```

### 2. Choose the Data Directory

Pass `--dir` on the command line — for example, to share your Mech Eye camera data:
```bash
python file_server.py --dir /home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/image_data_win
```

(You can still change the `SERVE_DIR` default at the top of `file_server.py` if you prefer.)

### 3. Transfer Files

#### Option A: Using the Client Script
```bash
# List files
python file_client.py list 192.168.1.100:8000

# Download a file
python file_client.py download 192.168.1.100:8000 filename.zip output.zip

# Upload a file
python file_client.py upload 192.168.1.100:8000 /path/to/file.zip
```

#### Option B: Using curl (from any computer)
```bash
# List files (JSON format)
curl http://192.168.1.100:8000/api/files

# Download a file
curl -O http://192.168.1.100:8000/path/to/file

# Upload a file
curl -X POST -H "X-Filename: myfile.zip" --data-binary @myfile.zip \
  http://192.168.1.100:8000/upload
```

#### Option C: Using Web Browser
Just navigate to `http://192.168.1.100:8000` to see a JSON listing of files.

## Features

✅ **Large file support** - Streams in 1MB chunks; 10GB+ files never load into RAM (verified)  
✅ **Concurrent transfers** - Threaded server; a large download won't block other clients  
✅ **Streaming downloads** - Download starts immediately, progress tracking  
✅ **Streaming uploads** - Upload large files without loading into memory  
✅ **Logging** - All requests/transfers logged to console and `file_server.log`  
✅ **Directory browsing** - Browse subdirectories, see file listings as JSON  
✅ **Security** - Path traversal protection built-in (returns 403)  
✅ **Cross-platform** - Works on Windows, macOS, Linux  

## Advanced Usage

### Resume an Interrupted Download
```bash
# With curl
curl -C - -O http://192.168.1.100:8000/largefile.zip

# With the client (requests library)
# Partial support - restart the download
```

### Monitor Transfer Speed
```bash
# Download with progress
curl -# -O http://192.168.1.100:8000/largefile.zip

# Or use the client script (shows live percentage)
python file_client.py download 192.168.1.100:8000 largefile.zip
```

### Change Server Port
```bash
python file_server.py --port 9000
```

## Troubleshooting

### Server starts but can't connect from another computer
- Check firewall: `sudo ufw allow 8000` (on Linux)
- Verify local network IP: `hostname -I`
- Check both computers are on the same network
- Try ping first: `ping 192.168.1.100`

### Slow transfer speeds
- Check network congestion: Run `iperf3` between machines
- Verify cable/Wi-Fi quality
- Try a different port if port 8000 is congested

### Large file transfers fail halfway
- Use curl with resume: `curl -C - -O http://server/file`
- Check disk space on destination
- Try uploading/downloading smaller test files first

### Permission denied errors
- Check file permissions: `ls -la`
- Ensure the SERVE_DIR is readable: `chmod 755 /path/to/dir`
- Run with sudo if needed (not recommended)

## Performance

Typical speeds on a local network:
- **Gigabit Ethernet**: 100+ MB/s
- **Wi-Fi 5 (802.11ac)**: 50-80 MB/s
- **Wi-Fi 6 (802.11ax)**: 100+ MB/s

10 GB transfer times:
- Gigabit Ethernet: ~100 seconds
- Wi-Fi 5: ~2-3 minutes
- Wi-Fi 6: ~1-2 minutes

## Security Notes

⚠️ **This server is for local networks only** - Do NOT expose to the internet

- The server runs in your LAN without encryption
- All files are accessible to anyone on the network
- For internet use, consider:
  - Running over VPN
  - Using HTTPS (requires setup)
  - Restricting access via firewall rules

## Requirements

- Python 3.6+
- `requests` library (for client script): `pip install requests`
- No other dependencies for the server

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.
