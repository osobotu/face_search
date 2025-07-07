from grpclib.client import Channel
from ahnlich_client_py.grpc.services import db_service

async def get_db_client(host="localhost", port=1369):
    channel = Channel(host=host, port=port)
    return db_service.DbServiceStub(channel), channel