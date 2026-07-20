with open("Mediscan/train_bone_age.py", "r", encoding="utf-8") as f:
    content = f.read()
content = (
    content.replace('"="*50', '"=" * 50').replace("len(bins)-1", "len(bins) - 1").replace("bins[i+1]", "bins[i + 1]")
)
with open("Mediscan/train_bone_age.py", "w", encoding="utf-8") as f:
    f.write(content)
