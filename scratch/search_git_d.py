import os


def search_for_git_d():
    search_roots = [r"D:\Program Files", r"D:\Program Files (x86)", r"D:\Git"]

    print("Searching for git.exe on D: drive...")
    found_paths = []

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "git.exe" in filenames:
                full_path = os.path.join(dirpath, "git.exe")
                print(f"Found: {full_path}")
                found_paths.append(full_path)
            if len(found_paths) >= 5:
                break

    if not found_paths:
        print("No git.exe found on D: drive in common roots.")
    else:
        print("Search complete. Found:", found_paths)


if __name__ == "__main__":
    search_for_git_d()
