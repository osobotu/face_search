import asyncio
from grpclib.client import Channel
from ahnlich_client_py.grpc.services.db_service import DbServiceStub
from ahnlich_client_py.grpc.db import query as db_query

async def ping(host: str = "127.0.0.1", port: int = 1369) -> str:
    async with Channel(host=host, port=port) as channel:
        db_client = DbServiceStub(channel)

        tracing_id = "00-80e1afed08e019fc1110464cfa66635c-7a085853722dc6d2-01"
        metadata = {"ahnlich-trace-id": tracing_id}

        try:
            response = await db_client.ping(db_query.Ping(), metadata=metadata)
            return f"Pong received: {response}"
        except Exception as e:
            return f"Failed to ping Ahnlich DB: {e}"

if __name__ == "__main__":
    print(asyncio.run(ping()))
