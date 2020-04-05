import numpy as np
import sqlite3
from timeit import timeit
import torchvision
from torch.utils.data import DataLoader, Subset
import os
import multiprocessing as mp
import litebatch as lb


def import_dataset(conn, table_name, dataset):
    conn.execute(
        f"""create table {table_name}(
        image_id integer primary key,
        image array,
        target integer)"""
    )
    for image, target in dataset:
        image = np.array(image)
        conn.execute(
            f"insert into {table_name} values (?,?,?)", (None, image, target),
        )


to_tensor = torchvision.transforms.ToTensor()


def transform(row):
    return to_tensor(row[0]), row[1]


if __name__ == "__main__":
    mp.set_start_method("spawn")

    num_workers = 16
    batch_size = 16
    tmp_dir = os.path.expanduser("~/tmp/")

    if not os.path.exists(tmp_dir + "mnist.db"):
        conn = sqlite3.connect(
            tmp_dir + "mnist.db", detect_types=sqlite3.PARSE_DECLTYPES
        )
        train_dataset = torchvision.datasets.MNIST(tmp_dir, train=True, download=True)
        import_dataset(conn, "train", train_dataset)
        conn.commit()
        conn.close()

    dataset = lb.LiteDataset(
        db_path=tmp_dir + "mnist.db",
        table="train",
        index="image_id",
        columns="image, target",
        transform=transform,
    )
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
