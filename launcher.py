import subprocess
import time
import webbrowser
import sys
import socket
import os


def is_port_open(port):
    """Checks if a TCP port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port_owner(port):
    """Terminates any process currently listening on the specified port (Windows-specific)."""
    if os.name != "nt":
        return
    try:
        # Find PID listening on port
        cmd = f"netstat -aon | findstr LISTENING | findstr :{port}"
        output = subprocess.check_output(cmd, shell=True).decode()
        for line in output.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                pid = parts[-1]
                print(f"[*] Port {port} is occupied. Terminating conflicting process ID: {pid}...")
                subprocess.call(
                    f"taskkill /F /PID {pid}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.5)
    except Exception:
        pass


def launch():
    print("=========================================================================")
    print("               BONE FRACTURE DETECTION & PROGNOSIS DASHBOARD")
    print("=========================================================================")
    print()

    # 1. Kill conflicting processes
    kill_port_owner(8000)

    # 2. Run Uvicorn backend in subprocess using current virtual environment python
    print("[*] Launching FastAPI App Server on http://127.0.0.1:8000/ ...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    # 3. Poll socket to wait for server binding (up to 40 tries, 20 seconds max)
    print("[*] Loading PyTorch deep learning weights into memory. Please wait...")
    server_ready = False
    for i in range(40):
        if is_port_open(8000):
            server_ready = True
            break
        time.sleep(0.5)

    if server_ready:
        print("[*] Server is active! Launching Web Dashboard in browser...")
        webbrowser.open("http://127.0.0.1:8000/")
    else:
        print("[!] Warning: Server did not bind to port 8000 within 20 seconds.")
        print("[!] Please inspect the console messages for PyTorch or database errors.")

    try:
        # Keep launcher alive to let user terminate using Ctrl+C
        process.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down application server...")
        process.terminate()


if __name__ == "__main__":
    launch()
