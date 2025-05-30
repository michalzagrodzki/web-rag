import asyncio
from prisma import Prisma
from config import settings

# Initialize Prisma client
prisma = Prisma()

async def connect_db():
    await prisma.connect()
    # Optional: test query
    now = await prisma.query_raw("SELECT NOW() AS current_time;")
    print("DB connected, current_time=", now)

async def disconnect_db():
    await prisma.disconnect()