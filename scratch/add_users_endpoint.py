
with open("d:/X-ray ML Model/main.py", "r", encoding="utf-8") as f:
    content = f.read()

endpoint_code = """
@app.get("/admin/users", tags=["Admin"])
def get_all_users(db: Session = Depends(get_db), current_user = Depends(get_current_admin)):
    users = db.query(db_models.User).all()
    return [{"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role, "is_active": u.is_active} for u in users]
"""

if '@app.get("/admin/users' not in content:
    content = content.replace("# Overrides", endpoint_code + "\n# Overrides")
    with open("d:/X-ray ML Model/main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Success: Added GET /admin/users to main.py")
else:
    print("Already exists")
