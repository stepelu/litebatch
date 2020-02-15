import numpy as np
from timeit import timeit
import litebatch as lb
import sqlite3

x = np.random.normal(size=[1, 28, 28]).astype(dtype=np.float32)

# Round-trip OK:
y = lb.deserialize(lb.serialize(x))
assert (x == y).all()

# SQLite3 registration OK:
con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("create table test (arr array)")
cur.execute("insert into test (arr) values (?)", (x,))
cur.execute("select arr from test")
y = cur.fetchone()[0]
assert (x == y).all()

# Test performance:
timeit_num = 100000
timeit_serialize = timeit(lambda: lb.deserialize(lb.serialize(x)), number=timeit_num)
print(f"array <-> serialized {timeit_num} times in {timeit_serialize:.3f}")
