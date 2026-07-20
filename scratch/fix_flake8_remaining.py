import os

# check_models.py
with open('check_models.py', 'r', encoding='utf-8') as f: content = f.read()
content = content.replace('import sys, os', 'import sys\nimport os').replace('import os, sys', 'import os\nimport sys')
with open('check_models.py', 'w', encoding='utf-8') as f: f.write(content)

# main.py
with open('main.py', 'r', encoding='utf-8') as f: content = f.read()
if 'logger = logging.getLogger' not in content:
    content = content.replace('import logging', 'import logging\nlogger = logging.getLogger(__name__)')
content = content.replace('from auth import UserCreate  # using db_models.User in endpoints\n', '')
content = content.replace('conn = sqlite3.connect("test.db")', 'sqlite3.connect("test.db")')
content = content.replace('except Exception as e:\n        return JSONResponse(', 'except Exception:\n        return JSONResponse(')
with open('main.py', 'w', encoding='utf-8') as f: f.write(content)

# Fix the F541 lines
files_to_fix = [
    'train_bone_age.py',
    'train_brain_hemorrhage.py',
    'train_retinopathy.py'
]
import re
for fname in files_to_fix:
    with open(fname, 'r', encoding='utf-8') as f: content = f.read()
    # just remove all 'f"' and "f'" where there's no brace
    content = re.sub(r'f(["\'])(.*?)\1', lambda m: m.group(0) if '{' in m.group(2) else m.group(1) + m.group(2) + m.group(1), content)
    with open(fname, 'w', encoding='utf-8') as f: f.write(content)

print("Fixes applied.")
