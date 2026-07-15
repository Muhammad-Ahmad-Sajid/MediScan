import subprocess

try:
    out = subprocess.check_output(
        "wmic process where \"name='python.exe'\" get ProcessID, CommandLine", shell=True
    ).decode("utf-8")
    print(out)
except Exception as e:
    print("Error:", e)
