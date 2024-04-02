import os
import datetime
from flask import Flask, render_template, redirect, url_for
# from database import DatabaseConnection
# from models import kayıt
import psycopg2 as p

try:
    conn = p.connect(dbname='ml', user='admin', host='localhost', password='postgres123', port=5433)
    conn.autocommit = True
    cur = conn.cursor()
    print("Successfully connected to the database!")
except p.OperationalError as e:
    print(f"Error: {e}")


