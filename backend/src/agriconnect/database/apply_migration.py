"""
Quick migration script: fix user_id column on conversations table.
Converts UUID→TEXT if needed, adds column if missing, creates index and FK.
"""
import sys
import psycopg2

def main():
    from backend.src.agriconnect.core.settings import settings
    db_url = settings.DATABASE_URL
    if not db_url:
        print("ERROR: DATABASE_URL not configured")
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    # 1. Check current state of user_id column
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'conversations' AND column_name = 'user_id'
    """)
    row = cur.fetchone()

    if row:
        print(f"Column user_id exists: type={row[1]}")
        if row[1] == "uuid":
            print("Converting UUID -> TEXT...")
            cur.execute("ALTER TABLE conversations ALTER COLUMN user_id TYPE TEXT USING user_id::TEXT")
            conn.commit()
            print("✅ Converted to TEXT")
        else:
            print("✅ Already TEXT, no change needed")
    else:
        print("Adding user_id column as TEXT...")
        cur.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
        conn.commit()
        print("✅ Added user_id TEXT column")

    # 2. Create index if missing
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
    conn.commit()
    print("✅ Index ensured")

    # 3. Try FK if possible
    try:
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'fk_conversations_user'
                ) THEN
                    ALTER TABLE conversations ADD CONSTRAINT fk_conversations_user
                        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE;
                END IF;
            END $$;
        """)
        conn.commit()
        print("✅ FK ensured")
    except Exception as e:
        conn.rollback()
        print(f"⚠️  FK skipped: {e}")

    # 4. Show final schema
    cur.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'conversations'
        ORDER BY ordinal_position
    """)
    print("\n=== conversations schema ===")
    for r in cur.fetchall():
        print(f"  {r[0]:30s} {r[1]:15s} nullable={r[2]}")

    cur.close()
    conn.close()
    print("\n✅ Migration complete")


if __name__ == "__main__":
    main()
