
import os
base = "d:/X-ray ML Model/Mediscan"
for root, dirs, files in os.walk(base):
    level = root.replace(base, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        for file in files[:2]:
            print(f"{chr(32)*2 * (level+1)}{file}")

