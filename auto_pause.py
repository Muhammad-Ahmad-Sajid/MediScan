import time
import subprocess
import datetime

# 7 hours in seconds
sleep_time = 7 * 60 * 60

log_file = "auto_pause.log"

# Log start
start_msg = (
    f"[{datetime.datetime.now().isoformat()}] Timer started. Will pause training in 7 hours ({sleep_time} seconds).\n"
)
print(start_msg)
with open(log_file, "a") as f:
    f.write(start_msg)

# Sleep for 7 hours
time.sleep(sleep_time)

# Execute PowerShell command to find and terminate the training python process
cmd = "powershell -Command \"Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe' AND CommandLine LIKE '%train_stage2.py%'\\\" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }\""
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# Log end
end_msg = f"[{datetime.datetime.now().isoformat()}] Auto-pause triggered. Output: {result.stdout.strip()} | Error: {result.stderr.strip()}\n"
print(end_msg)
with open(log_file, "a") as f:
    f.write(end_msg)
