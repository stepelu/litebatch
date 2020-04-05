from msgpack import packb, unpackb
from msgpack_numpy import encode as msgp_encode
from msgpack_numpy import decode as msgp_decode


def encode(obj):
    return packb(obj, default=msgp_encode)


def decode(bytestr):
    return unpackb(bytestr, object_hook=msgp_decode)
