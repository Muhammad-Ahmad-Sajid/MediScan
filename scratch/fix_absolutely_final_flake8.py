with open("main.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "with engine.connect() as conn:" in line:
        lines[i] = line.replace(
            "with engine.connect() as conn:", "with engine.connect():"
        )
    if 'raise HTTPException(500, f"Inference failed: {str(e)}")' in line:
        lines[i] = line.replace('f"Inference failed: {str(e)}"', '"Inference failed"')
    if 'logger.error(f"Retinopathy inference failed: ", exc_info=True)' in line:
        lines[i] = line.replace(
            'f"Retinopathy inference failed: "', '"Retinopathy inference failed: "'
        )

with open("main.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Fixes applied.")
