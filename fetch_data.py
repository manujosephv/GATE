import gzip
import os
import pickle
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from category_encoders import LeaveOneOutEncoder  # noqa: F401
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import (  # noqa: F401
    StratifiedShuffleSplit,
    train_test_split,
)
from tqdm import tqdm


def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """saves file from url to filename with a fancy progressbar"""
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get("content-length")

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length, position=0, leave=True) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))
    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename


def fetch_A9A(path, train_size=None, valid_size=None, test_size=None):
    os.makedirs(path, exist_ok=True)
    train_path = os.path.join(path, "a9a")
    test_path = os.path.join(path, "a9a.t")
    if not all(os.path.exists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/w09tcm95834l3eg/a9a?dl=1", train_path)
        download("https://www.dropbox.com/s/659khq0hih4krwm/a9a.t?dl=1", test_path)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=123)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=123)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = os.path.join(path, "stratified_train_idx.txt")
        valid_idx_path = os.path.join(path, "stratified_valid_idx.txt")
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download(
                "https://www.dropbox.com/s/q41rx03mhistxxx/stratified_train_idx.txt?dl=1",
                train_idx_path,
            )
            download(
                "https://www.dropbox.com/s/3eu7t9507p4n1vy/stratified_valid_idx.txt?dl=1",
                valid_idx_path,
            )
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn(
                "train_size + valid_size = {} exceeds dataset size: {}.".format(
                    train_size + valid_size, len(X_train)
                ),
                Warning,
            )
        if test_size is not None:
            warnings.warn("Test set is fixed for this dataset.", Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size : train_size + valid_size]

    return dict(
        X_train=X_train[train_idx],
        y_train=y_train[train_idx],
        X_valid=X_train[valid_idx],
        y_valid=y_train[valid_idx],
        X_test=X_test,
        y_test=y_test,
        num_features=X_train.shape[1],
        num_classes=2,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
    )


def fetch_YEAR(path, train_size=None, valid_size=None, test_size=51630):
    os.makedirs(path, exist_ok=True)
    pkl_path = f"{path}/YEAR_set_1_.pickle"
    if os.path.isfile(pkl_path):
        print("====== fetch_YEAR@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
        # print(f"====== fetch_YEAR:\tX_={data_dict['X'].shape}\tY={data_dict['Y'].shape}")
    else:
        data_path = os.path.join(path, "data.csv")
        if not os.path.exists(data_path):
            os.makedirs(path, exist_ok=True)
            download(
                "https://www.dropbox.com/s/jmlxwez036jz0b9/YearPredictionMSD.txt?dl=1",
                data_path,
            )
        n_features = 91
        types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
        data = pd.read_csv(data_path, header=None, dtype=types)
        if False:
            data_dict = {"X": data.iloc[:, 1:].values, "Y": data.iloc[:, 0].values}
        else:
            data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

            X_train, y_train = (
                data_train.iloc[:, 1:].values,
                data_train.iloc[:, 0].values,
            )
            X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

            if all(sizes is None for sizes in (train_size, valid_size)):
                train_idx_path = os.path.join(path, "stratified_train_idx.txt")
                valid_idx_path = os.path.join(path, "stratified_valid_idx.txt")
                if not all(
                    os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)
                ):
                    download(
                        "https://www.dropbox.com/s/2kwy8jwrhaqmo0w/stratified_train_idx.txt?dl=1",
                        train_idx_path,
                    )
                    download(
                        "https://www.dropbox.com/s/vk7qqa2phdpg1y9/stratified_valid_idx.txt?dl=1",
                        valid_idx_path,
                    )
                train_idx = pd.read_csv(train_idx_path, header=None)[0].values
                valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
            else:
                assert train_size, "please provide either train_size or none of sizes"
                if valid_size is None:
                    valid_size = len(X_train) - train_size
                    assert valid_size > 0
                if train_size + valid_size > len(X_train):
                    warnings.warn(
                        "train_size + valid_size = {} exceeds dataset size: {}.".format(
                            train_size + valid_size, len(X_train)
                        ),
                        Warning,
                    )

                shuffled_indices = np.random.permutation(np.arange(len(X_train)))
                train_idx = shuffled_indices[:train_size]
                valid_idx = shuffled_indices[train_size : train_size + valid_size]
            print(
                f"fetch_YEAR\ttrain={X_train[train_idx].shape} valid={X_train[valid_idx].shape} test={X_test.shape}"
            )
            data_dict = dict(
                X_train=X_train[train_idx],
                y_train=y_train[train_idx],
                X_valid=X_train[valid_idx],
                y_valid=y_train[valid_idx],
                X_test=X_test,
                y_test=y_test,
                num_features=X_train.shape[1],
                feature_names=["f{}".format(i) for i in range(X_train.shape[1])],
            )
            with open(pkl_path, "wb") as fp:
                pickle.dump(data_dict, fp)
        # print(f"====== fetch_HIGGS:\tX={data_dict['X'].shape}")
    print(
        f"====== fetch_YEAR:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}"
        f"\tX_test={data_dict['X_test'].shape}"
    )
    return data_dict


def fetch_MICROSOFT(path):
    os.makedirs(path, exist_ok=True)
    pkl_path = f"{path}/MICROSOFT_set_1_.pickle"
    if os.path.isfile(pkl_path):
        print("====== fetch_MICROSOFT@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
        # X_train=(580539, 0)	X_valid=(142873, 0)	X_test=(241521, 0)
        print(
            f"====== fetch_MICROSOFT:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}\tX_test={data_dict['X_test'].shape}"
        )
    else:
        train_path = os.path.join(path, "msrank_train.tsv")
        test_path = os.path.join(path, "msrank_test.tsv")
        if not all(os.path.exists(fname) for fname in (train_path, test_path)):
            os.makedirs(path, exist_ok=True)
            download(
                "https://www.dropbox.com/s/idu4ierw5e2emj7/msrank_train.tsv?dl=1",
                train_path,
            )
            download(
                "https://www.dropbox.com/s/b8o2pygq9e1o9er/msrank_test.tsv?dl=1",
                test_path,
            )

        for fname in (train_path, test_path):
            raw = open(fname).read().replace("\\t", "\t")
            with open(fname, "w") as f:
                f.write(raw)

        data_train = pd.read_csv(train_path, header=None, skiprows=1, sep="\t")
        data_test = pd.read_csv(test_path, header=None, skiprows=1, sep="\t")

        train_idx_path = os.path.join(path, "train_idx.txt")
        valid_idx_path = os.path.join(path, "valid_idx.txt")
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download(
                "https://www.dropbox.com/s/ei67bllm4jtqz2s/train_idx.txt?dl=1",
                train_idx_path,
            )
            download(
                "https://www.dropbox.com/s/chgil277fwn60pg/valid_idx.txt?dl=1",
                valid_idx_path,
            )
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values

        X_train, y_train, query_train = (
            data_train.iloc[train_idx, 2:].values,
            data_train.iloc[train_idx, 0].values,
            data_train.iloc[train_idx, 1].values,
        )
        X_valid, y_valid, query_valid = (
            data_train.iloc[valid_idx, 2:].values,
            data_train.iloc[valid_idx, 0].values,
            data_train.iloc[valid_idx, 1].values,
        )
        X_test, y_test, query_test = (
            data_test.iloc[:, 2:].values,
            data_test.iloc[:, 0].values,
            data_test.iloc[:, 1].values,
        )

        data_dict = dict(
            X_train=X_train.astype(np.float32),
            y_train=y_train.astype(np.int64),
            query_train=query_train,
            X_valid=X_valid.astype(np.float32),
            y_valid=y_valid.astype(np.int64),
            query_valid=query_valid,
            X_test=X_test.astype(np.float32),
            y_test=y_test.astype(np.int64),
            query_test=query_test,
            num_features=X_train.shape[1],
            feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
        )
        print(
            f"====== fetch_MICROSOFT:\tX_train={X_train.shape}\tX_valid={X_valid.shape}\tX_test={X_test.shape}"
        )
        with open(pkl_path, "wb") as fp:
            pickle.dump(data_dict, fp)
    return data_dict


def fetch_CLICK(path, valid_size=100_000, validation_seed=None):
    os.makedirs(path, exist_ok=True)
    pkl_path = f"{path}/click_set_1_.pickle"
    if not os.path.isfile(pkl_path):
        print("====== Downloading Click Pickle ......")
        download(
            "https://www.dropbox.com/s/ry6zsr6qtuz8l5z/click_set_1_.pickle?dl=1",
            pkl_path,
        )
    print("====== fetch_CLICK@{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        data_dict = pickle.load(fp)
    # else:
    #     download(
    #             "https://www.dropbox.com/s/q41rx03mhistxxx/stratified_train_idx.txt?dl=1",
    #             pkl_path,
    #         )
    #     # based on: https://www.kaggle.com/slamnz/primer-airlines-delay
    #     csv_path = os.path.join(path, "click.csv")
    #     if not os.path.exists(csv_path):
    #         os.makedirs(path, exist_ok=True)
    #         download(
    #             "https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1", csv_path
    #         )

    #     data = pd.read_csv(csv_path, index_col=0)
    #     X, y = data.drop(columns=["target"]), data["target"]
    #     X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
    #     y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

    #     y_train = (y_train.values.reshape(-1) == 1).astype("int64")
    #     y_test = (y_test.values.reshape(-1) == 1).astype("int64")

    #     cat_features = [
    #         "url_hash",
    #         "ad_id",
    #         "advertiser_id",
    #         "query_id",
    #         "keyword_id",
    #         "title_id",
    #         "description_id",
    #         "user_id",
    #     ]

    #     X_train, X_val, y_train, y_val = train_test_split(
    #         X_train, y_train, test_size=valid_size, random_state=validation_seed
    #     )

    #     num_features = X_train.shape[1]
    #     num_classes = len(set(y_train))
    #     cat_encoder = LeaveOneOutEncoder()
    #     cat_encoder.fit(X_train[cat_features], y_train)
    #     X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    #     X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
    #     X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
    #     data_dict = dict(
    #         X_train=X_train.values.astype("float32"),
    #         y_train=y_train,
    #         X_valid=X_val.values.astype("float32"),
    #         y_valid=y_val,
    #         X_test=X_test.values.astype("float32"),
    #         y_test=y_test,
    #         num_features=num_features,
    #         num_classes=num_classes,
    #         feature_names=data.columns.tolist(),
    #     )
    #     # print(f"====== fetch_CLICK:\tX_train={X_train.shape}\tX_valid={X_valid.shape}\tX_test={X_test.shape}")
    #     with open(pkl_path, "wb") as fp:
    #         pickle.dump(data_dict, fp)
    print(
        f"====== fetch_CLICK:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}"
        f"\tX_test={data_dict['X_test'].shape}"
    )
    return data_dict


def fetch_FOREST(path, train_size=None, valid_size=5 * 10**4, test_size=5 * 10**4):
    os.makedirs(path, exist_ok=True)
    pkl_path = f"{path}/FOREST_set_1_.pickle"
    if os.path.isfile(pkl_path):
        print("====== fetch_FOREST@{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            data_dict = pickle.load(fp)
        print(
            f"====== fetch_FOREST:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}"
            f"\tX_test={data_dict['X_test'].shape}"
        )
    else:
        data_path = os.path.join(path, "forest.csv")

        if not os.path.exists(data_path):
            os.makedirs(path, exist_ok=True)
            archive_path = os.path.join(path, "covtype.data.gz")
            download(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                archive_path,
            )
            with gzip.open(archive_path, "rb") as f_in:
                with open(data_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        feature_name_n = [
            "elevation",
            "aspect",
            "slope",
            "h_dist_hydrology",
            "v_dist_hydrology",
            "h_dist_roadways",
            "hillshade_9am",
            "hillshade_noon",
            "hillshade_3pm",
            "h_dist_firepoints",
        ]
        feature_name_oh = [f"wilderness_area_{i}" for i in range(4)] + [
            f"soil_type_{i}" for i in range(40)
        ]
        feature_names = feature_name_n + feature_name_oh
        types = {
            i: (np.float32 if f in feature_name_n else np.int)
            for i, f in enumerate(feature_names)
        }
        data = pd.read_csv(data_path, dtype=types, names=feature_names + ["Cover_Type"])
        num_features = len(feature_names)
        Y = data.loc[:, "Cover_Type"].values
        num_classes = len(set(Y))
        X, y = (
            data.iloc[:, :-1].values,
            data.iloc[:, -1].values,
        )
        train_idx_path = os.path.join(path, "stratified_train_idx.txt")
        valid_idx_path = os.path.join(path, "stratified_valid_idx.txt")
        test_idx_path = os.path.join(path, "stratified_test_idx.txt")
        if True:
            if not all(
                os.path.exists(fname)
                for fname in (train_idx_path, valid_idx_path, test_idx_path)
            ):
                download(
                    "https://www.dropbox.com/s/g65thgcy4f8xd97/stratified_train_idx.txt?dl=1",
                    train_idx_path,
                )
                download(
                    "https://www.dropbox.com/s/6e31cam7dm9ws3s/stratified_valid_idx.txt?dl=1",
                    valid_idx_path,
                )
                download(
                    "https://www.dropbox.com/s/50182xek5n44aiz/stratified_test_idx.txt?dl=1",
                    test_idx_path,
                )
            train_idx = pd.read_csv(train_idx_path, header=None)[0].values
            valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
            test_idx = pd.read_csv(test_idx_path, header=None)[0].values
        else:
            X_indices = np.arange(len(X))
            # Only first time to generate the train valid test split
            splitter = StratifiedShuffleSplit(
                n_splits=1, random_state=0, test_size=test_size
            )
            _train_idx, _test_idx = list(splitter.split(X_indices, y))[0]
            test_idx = X_indices[_test_idx]

            train_indices = X_indices[_train_idx]
            y_train_indices = y[_train_idx]
            splitter = StratifiedShuffleSplit(
                n_splits=1, random_state=0, test_size=valid_size
            )
            _train_idx, _valid_idx = list(
                splitter.split(train_indices, y_train_indices)
            )[0]
            train_idx = train_indices[_train_idx]
            valid_idx = train_indices[_valid_idx]
            np.savetxt(train_idx_path, train_idx, fmt="%d")
            np.savetxt(valid_idx_path, valid_idx, fmt="%d")
            np.savetxt(test_idx_path, test_idx, fmt="%d")

        X_train, y_train = (
            X[train_idx],
            y[train_idx],
        )
        X_valid, y_valid = (
            X[valid_idx],
            y[valid_idx],
        )
        X_test, y_test = (
            X[test_idx],
            y[test_idx],
        )
        data_dict = dict(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            num_features=num_features,
            num_classes=num_classes,
            feature_names=feature_names,
        )
        with open(pkl_path, "wb") as fp:
            pickle.dump(data_dict, fp)
    print(
        f"====== fetch_FOREST:\tX_train={data_dict['X_train'].shape}\tX_valid={data_dict['X_valid'].shape}"
        f"\tX_test={data_dict['X_test'].shape}"
    )
    return data_dict


DATASETS = {
    "A9A": fetch_A9A,
    "YEAR": fetch_YEAR,
    "MICROSOFT": fetch_MICROSOFT,
    "FOREST": fetch_FOREST,
    "CLICK": fetch_CLICK,
}
# import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("root_path", type=str, help="root path to save data", default="datasets")
    # args = parser.parse_args()
    # root_path = Path(args.root_path)
    root_path = Path("datasets")
    os.makedirs(root_path, exist_ok=True)
    fetch_CLICK(root_path / "click")
    fetch_YEAR(root_path / "year")
    fetch_MICROSOFT(root_path / "microsoft")
    fetch_FOREST(root_path / "forest")
    fetch_A9A(root_path / "a9a")
