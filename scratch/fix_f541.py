import glob
import re

for filename in glob.glob('**/*.py', recursive=True):
    if 'venv' in filename or 'myenv' in filename: continue
    with open(filename, 'r', encoding='utf-8') as f: content = f.read()
    
    # Replace 'some string' with 'some string' and "some string" with "some string"
    # Only if the string doesn't contain '{' or '}'
    content = re.sub(r'f(["\'])([^"{}\'\\]+)\1', r'\1\2\1', content)
    
    with open(filename, 'w', encoding='utf-8') as f: f.write(content)
print("Done fixing F541")
