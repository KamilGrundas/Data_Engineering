import json
from pathlib import Path
from typing import Any
from decimal import Decimal
import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
DB_NAME = os.getenv("PGDATABASE", "library_db")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD", "postgres")

def load_data(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()
    return text

def fix_json_data(text: str) -> str:
    keys = [":id", ":title", ":author", ":genre", ":publisher", ":year", ":price"]
    fixed_text = text.replace("=>",":")
    for key in keys:
        fixed_text = fixed_text.replace(f'{key}', f'"{key.replace(":", "")}"')
    
    return fixed_text

def save_fixed_json_file(path: Path, load_json) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(load_json, file, indent=4, ensure_ascii=False)

def normalize_price(price: str) -> Decimal:
    return Decimal(price.replace('$', '').replace("€", ""))

def json_to_rows(load_json: Any) -> list[tuple]:
    books = []
    for book in load_json:
        row = (
            str(book['id']), # id number is too big for BIGINT
            book['title'],
            book['author'],
            book['genre'],
            book['publisher'],
            int(book['year']),
            normalize_price(book['price'])
        )
        books.append(row)
    return books

def ensure_table(conn):
    create_sql = """
    CREATE TABLE IF NOT EXISTS books (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT,
        genre TEXT,
        publisher TEXT,
        year INTEGER,
        price NUMERIC(10,2)
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)

def upsert_books(conn, rows):
    """
    Wstawia lub aktualizuje rekordy. Używamy executemany — prosty i czytelny.
    """
    insert_sql = """
    INSERT INTO books (id, title, author, genre, publisher, year, price)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE
    SET title = EXCLUDED.title,
        author = EXCLUDED.author,
        genre = EXCLUDED.genre,
        publisher = EXCLUDED.publisher,
        year = EXCLUDED.year,
        price = EXCLUDED.price;
    """
    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)


if __name__ == "__main__":
    FILE = Path(__file__).parent / "task1_d.json"
    FIXED_FILE = Path(__file__).parent / "task1_fixed.json"
    raw_text = load_data(FILE)
    fixed_text = fix_json_data(raw_text)
    load_json = json.loads(fixed_text)
    save_fixed_json_file(FIXED_FILE, load_json)
    books = json_to_rows(load_json)
    # for book in books:
    #     if book[0] > 9223372036854775807:
    #         raise ValueError(f"ID {book[0]} exceeds BIGINT limit.")
    conninfo = {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
    }
    with psycopg.connect(**conninfo) as conn: # type: ignore
        ensure_table(conn)
        upsert_books(conn, books)
        conn.commit()
        print("Wstawiono/aktualizowano wiersze.")

