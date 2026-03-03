#!/usr/bin/env python3
"""Create an admin user. Usage: python create_admin.py <username> <password>"""
import sys
from db import init_db, get_pg
from auth import hash_password

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_admin.py <username> <password>")
        sys.exit(1)
    username, password = sys.argv[1], sys.argv[2]
    init_db()
    conn = get_pg()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO admin_users (username, password_hash) VALUES (%s, %s) ON CONFLICT (username) DO UPDATE SET password_hash=EXCLUDED.password_hash",
        (username, hash_password(password)))
    cur.close()
    print(f"Admin user '{username}' created/updated.")
