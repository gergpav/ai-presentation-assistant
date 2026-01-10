import asyncio
from app.db.session import AsyncSessionLocal
from app.db.models.user import User

async def main():
    async with AsyncSessionLocal() as session:
        user = User(login="test1", password_hash="hash")
        session.add(user)
        await session.commit()
        print("OK, user id:", user.id)

asyncio.run(main())
