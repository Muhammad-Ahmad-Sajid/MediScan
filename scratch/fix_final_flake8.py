import re

# Fix check_models.py
with open("check_models.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace(
    "import json, torch, os", "import json\nimport torch\nimport os"
)
with open("check_models.py", "w", encoding="utf-8") as f:
    f.write(content)

# Fix main.py
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix F841 local variable 'e' and F541 missing placeholders
content = re.sub(
    r'except ImportError as e:\n    logger.warning\(f"([^"]+)"\)',
    r'except ImportError:\n    logger.warning("\1")',
    content,
)

# Fix F841 local variable 'conn'
content = content.replace("conn = sqlite3.connect", "sqlite3.connect")

# Fix F821 undefined name 'e' and F541 missing placeholders
content = re.sub(
    r'except Exception:\n        logger.error\(f"([^"]+)"\)',
    r'except Exception:\n        logger.error("\1")',
    content,
)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixes applied successfully.")
