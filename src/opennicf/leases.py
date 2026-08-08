"""SQLite-backed, expiring execution leases."""
import sqlite3, time

class LeaseStore:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS leases(issue INTEGER PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL)")
        self.db.commit()

    def acquire(self, issue: int, owner: str, ttl: float = 900) -> bool:
        now = time.time()
        self.db.execute("DELETE FROM leases WHERE expires <= ?", (now,))
        try:
            self.db.execute("INSERT INTO leases VALUES (?, ?, ?)", (issue, owner, now + ttl))
        except sqlite3.IntegrityError:
            self.db.rollback()
            return False
        self.db.commit()
        return True

