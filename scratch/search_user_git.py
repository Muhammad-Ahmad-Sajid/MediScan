import os


def search_user_git():
    search_root = r"C:\Users\HP"
    print("Searching C:\\Users\\HP for git.exe...")
    found_paths = []

    # We walk the user directory but skip common heavy directories to save time
    skip_dirs = {
        "venv",
        ".venv",
        "env",
        "node_modules",
        ".git",
        "AppData\\Local\\Temp",
        "Downloads",
        "Pictures",
        "Videos",
        "Music",
        "3D Objects",
    }

    for dirpath, dirnames, filenames in os.walk(search_root):
        # Filter out skip directories in place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]

        if "git.exe" in filenames:
            full_path = os.path.join(dirpath, "git.exe")
            print(f"Found: {full_path}")
            found_paths.append(full_path)

    if not found_paths:
        print("No git.exe found inside C:\\Users\\HP.")
    else:
        print("Search complete. Found:", found_paths)


if __name__ == "__main__":
    search_user_git()
