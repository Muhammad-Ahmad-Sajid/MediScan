from database import SessionLocal
from auth import User, hash_password

db = SessionLocal()
user = db.query(User).filter(User.email == "admin@example.com").first()
if user:
    user.hashed_password = hash_password("password")
else:
    user = User(
        full_name="Admin",
        email="admin@example.com",
        hashed_password=hash_password("password"),
        role="admin",
        is_active=True,
    )
    db.add(user)
db.commit()
print("Successfully fixed password hash for admin user!")
