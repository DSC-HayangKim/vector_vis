import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv


class MongoDatabase:
    def __init__(self, database_name : str):
        load_dotenv()
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "27017")
        self.mongo_uri : str = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/"
        self.database_name = database_name

    def get_database(self):
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.database_name]
        return self.db
    
    def insert_document(self, collection_name : str, document : dict):
        self.get_database()
        collection = self.db[collection_name]

        result = collection.insert_one(document)
        return result.inserted_id
