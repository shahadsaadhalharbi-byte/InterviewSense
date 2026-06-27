import sqlite3

# Connect to (or create) the database
conn = sqlite3.connect('database.db')

# Create the users table
conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        major TEXT
    )
''')
conn.commit()

print("✅ Database and users table created successfully.")
conn.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        message TEXT NOT NULL
    )
''')
conn.commit()
conn.close()

print("✅ contacts table created.")
