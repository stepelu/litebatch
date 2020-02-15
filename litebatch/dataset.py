import sqlite3
import numpy as np
from litebatch.serialization import serialize, deserialize


sqlite3.register_adapter(np.ndarray, serialize)
sqlite3.register_converter("array", deserialize)


def no_transform(row):
    return row


class LiteDataset:
    def __init__(self, db_name, len_query, select_query, transform=no_transform):
        super().__init__()
        self.db_name = db_name
        self.len_query = len_query
        self.get_query = select_query
        self.transform = transform
        self.conn = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.c = self.conn.cursor()

    def __len__(self):
        return self.c.execute(self.len_query).fetchone()[0]

    def __getitem__(self, i):
        row = self.c.execute(self.get_query, [i]).fetchone()
        return self.transform(row)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["conn"]
        del state["c"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.c = self.conn.cursor()
