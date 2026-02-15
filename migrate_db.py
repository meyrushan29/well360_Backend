
import os
import psycopg2
from pathlib import Path

def migrate():
    """
    Manually add missing columns to the lip_analysis table.
    Base.metadata.create_all() does not add columns to existing tables.
    """
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("MIGRATION: No DATABASE_URL found. Skipping auto-migration.")
        return

    # Handle 'postgres://' vs 'postgresql://' for psycopg2
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        print("MIGRATION: Checking 'lip_analysis' table for missing columns...")
        
        # Add xai_url if missing
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='lip_analysis' AND column_name='xai_url') THEN
                    ALTER TABLE lip_analysis ADD COLUMN xai_url VARCHAR;
                    RAISE NOTICE 'Added column xai_url';
                END IF;
            END $$;
        """)
        
        # Add xai_description if missing
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='lip_analysis' AND column_name='xai_description') THEN
                    ALTER TABLE lip_analysis ADD COLUMN xai_description TEXT;
                    RAISE NOTICE 'Added column xai_description';
                END IF;
            END $$;
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("MIGRATION: Database successfully updated.")
        
    except Exception as e:
        print(f"MIGRATION ERROR: {e}")

if __name__ == "__main__":
    migrate()
