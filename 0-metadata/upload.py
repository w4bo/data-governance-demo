from pymongo import MongoClient
import json, glob, os

# uri = os.getenv("MONGO_URI", "mongodb://mongodb:27017/db")
client = MongoClient("mongodb://137.204.74.36:27017/db")
db = client.get_default_database()
collection = db["photo-gallery"]

# clean existing
collection.drop()

for file in glob.glob("/home/0-metadata/out/*.json"):
    with open(file, "r") as f:
        data = json.load(f)
        collection.insert_one(data)

print("✅ All JSON files inserted")
