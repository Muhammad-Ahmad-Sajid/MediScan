import os


def search_for_git():
    search_roots = [r"C:\Program Files", r"C:\Program Files (x86)", r"C:\Users\HP\AppData"]

    print("Searching for git.exe on disk...")
    found_paths = []

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "git.exe" in filenames:
                full_path = os.path.join(dirpath, "git.exe")
                print(f"Found: {full_path}")
                found_paths.append(full_path)
                # Keep searching to see if there are multiple
            # Avoid walking deep into unrelated folders to speed up
            if len(found_paths) >= 5:
                break

    if not found_paths:
        print("No git.exe found on disk in common roots.")
    else:
        print("Search complete. Found:", found_paths)


if __name__ == "__main__":
    search_for_git()
