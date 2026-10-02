from pymongo import MongoClient
import json, glob, os
from dotenv import load_dotenv

# Load .env from the parent folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

mongo_host = os.getenv("MONGO_HOST", os.getenv("IP", "mongodb"))
mongo_uri = f"mongodb://{mongo_host}:27017/db"
if not mongo_uri:
    raise ValueError("MONGO_URI not set in .env")

client = MongoClient(mongo_uri)
db = client.get_default_database()
collection = db["photo-gallery"]

# clean existing
collection.drop()

for file in glob.glob("/home/0-metadata/out/*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        collection.insert_one(data)
