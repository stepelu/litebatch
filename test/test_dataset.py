import numpy as np
import sqlite3
from timeit import timeit
import torchvision
from torch.utils.data import DataLoader, Subset
import os
import multiprocessing as mp
import litebatch as lb


def create_db(db_name, table_name, image_shape, dataset):
    # TODO: Move to litebatch?
    conn = sqlite3.connect(db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(
        f"""create table
        {table_name}(row_id integer primary key, image array, target integer)"""
    )
    for image, target in enumerate(dataset):
        image = np.array(image).reshape(image_shape)
        c.execute(f"insert into {table_name} values (?, ?, ?)", (None, image, target))
    conn.commit()
    conn.close()


to_tensor = torchvision.transforms.ToTensor()


def transform(row):
    return to_tensor(row[0]), row[1]


if __name__ == "__main__":
    mp.set_start_method("spawn")

    num_workers = 16
    batch_size = 16
    tmp_dir = os.path.expanduser("~/tmp/")

    if not os.path.exists(tmp_dir + "mnist.db"):
        train_dataset = torchvision.datasets.MNIST(tmp_dir, train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(tmp_dir, train=False, download=True)
        create_db(tmp_dir + "mnist.db", "train", (1, 28, 28), train_dataset)
        create_db(tmp_dir + "mnist.db", "test", (1, 28, 28), test_dataset)

    dataset = lb.LiteDataset(
        db_path=tmp_dir + "mnist.db",
        table="train",
        index="row_id",
        columns="image, target",
        transform=transform,
    )
    print(dataset.select_query)
    indices = dataset.cursor.execute("select row_id from train").fetchall()
    indices = [i[0] for i in indices]
    dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    print(timeit(lambda: [[x, y] for x, y in loader], number=1))

    loader = DataLoader(
        torchvision.datasets.MNIST(tmp_dir + "mnist", transform=to_tensor),
        num_workers=num_workers,
        batch_size=batch_size,
    )
    print(timeit(lambda: [[x, y] for x, y in loader], number=1))
