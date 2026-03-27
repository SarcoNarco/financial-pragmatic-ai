import os

from motor.motor_asyncio import AsyncIOMotorClient


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "financial_ai")

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]

users_collection = db["users"]
transcripts_collection = db["transcripts"]
analyses_collection = db["analyses"]


async def init_database():
    await users_collection.create_index("email", unique=True)
    await analyses_collection.create_index("user_id")
    await analyses_collection.create_index("created_at")
    await transcripts_collection.create_index("user_id")
