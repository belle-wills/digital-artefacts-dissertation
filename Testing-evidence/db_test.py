from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:fitness123@localhost:5432/fitness_ai"

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1")).scalar()
    print("DB connected, SELECT 1 ->", result)