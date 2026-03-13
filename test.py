from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:barca%40123@localhost:5432/Zora")

conn = engine.connect()

print("Connected successfully!")