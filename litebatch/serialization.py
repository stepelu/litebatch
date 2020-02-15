import msgpack
import msgpack_numpy


def serialize(obj):
    return msgpack.packb(obj, default=msgpack_numpy.encode)


def deserialize(data):
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
