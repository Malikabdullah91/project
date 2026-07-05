from app import app, db, User

with app.app_context():
    admin = User(
        username="admin",
        email="admin@email.com",
        password="password123" 
    )
    db.session.add(admin)
    db.session.commit()
    print("Admin berhasil ditambahkan!")