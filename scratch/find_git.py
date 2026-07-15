import subprocess


def find_git():
    possible_paths = [
        "git",
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
        r"C:\Users\HP\AppData\Local\Programs\Git\cmd\git.exe",
    ]
    for path in possible_paths:
        try:
            res = subprocess.run([path, "--version"], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"FOUND_GIT:{path}")
                return path
        except Exception:
            continue
    print("FOUND_GIT:None")
    return None


if __name__ == "__main__":
    find_git()
