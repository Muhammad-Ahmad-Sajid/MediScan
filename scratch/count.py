
import os

base = "d:/X-ray ML Model/Mediscan"
splits = ["train", "val", "test", "auto_test"]
grades = ["0", "1", "2", "3", "4"]
grade_names = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]

print(f"{'Split':<12} {'Grade':<6} {'Name':<12} {'Count':<8}")
print("-" * 40)
total = 0
for split in splits:
    for grade, name in zip(grades, grade_names):
        path = os.path.join(base, split, grade)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith('.png')])
            total += count
            print(f"{split:<12} {grade:<6} {name:<12} {count:<8}")
    print()
print(f"Total images: {total}")

