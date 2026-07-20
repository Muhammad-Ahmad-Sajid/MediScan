import os

svg_dir = r"d:\X-ray ML Model\static\img\module-icons"
for file in os.listdir(svg_dir):
    if file.endswith(".svg"):
        filepath = os.path.join(svg_dir, file)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if 'xmlns="http://www.w3.org/2000/svg"' not in content:
            content = content.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ')
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {file}")
