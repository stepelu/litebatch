import sqlite3
import numpy as np
from litebatch.serialization import encode, decode


sqlite3.register_adapter(np.ndarray, encode)
sqlite3.register_converter("array", decode)


class LiteDataset:
    def __init__(self, db_path, table, index, columns, transform=None):
        super().__init__()
        self.db_path = db_path
        self.select_query = f"select {columns} from {table} where {index}=?"
        self.transform = transform
        self._connect()

    def _connect(self):
        self.connection = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self.cursor = self.connection.cursor()

    def __getitem__(self, i):
        row = self.cursor.execute(self.select_query, [int(i)]).fetchone()
        return self.transform(row) if self.transform is not None else row

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["connection"]
        del state["cursor"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._connect()
