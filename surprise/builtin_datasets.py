"""This module contains built-in datasets that can be automatically
downloaded."""

import errno
import os
import ssl
import zipfile
from collections import namedtuple
from os.path import join
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve


def get_dataset_dir():
    """Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.surprise_data/, but it can also be set by the
    environment variable ``SURPRISE_DATA_FOLDER``.
    """

    folder = os.environ.get(
        "SURPRISE_DATA_FOLDER", os.path.expanduser("~") + "/.surprise_data/"
    )
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            # reraise exception if folder does not exist and creation failed.
            raise

    return folder


# a builtin dataset has
# - an url (where to download it)
# - a relative path (relative to dataset directory)
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple("BuiltinDataset", ["url", "relative_path", "reader_params"])


def get_builtin_datasets():
    """Return builtin datasets with dynamic paths based on current data directory."""
    return {
        "ml-100k": BuiltinDataset(
            url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            relative_path="ml-100k/ml-100k/u.data",
            reader_params=dict(
                line_format="user item rating timestamp", rating_scale=(1, 5), sep="\t"
            ),
        ),
        "ml-1m": BuiltinDataset(
            url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            relative_path="ml-1m/ml-1m/ratings.dat",
            reader_params=dict(
                line_format="user item rating timestamp", rating_scale=(1, 5), sep="::"
            ),
        ),
        "jester": BuiltinDataset(
            url="https://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip",
            relative_path="jester/jester_ratings.dat",
            reader_params=dict(line_format="user item rating", rating_scale=(-10, 10)),
        ),
    }

# Keep the old BUILTIN_DATASETS for backward compatibility, but it uses dynamic paths
# This creates a property-like object that always returns current datasets


class _BuiltinDatasetsProxy:
    def __getitem__(self, key):
        return get_builtin_datasets()[key]

    def __contains__(self, key):
        return key in get_builtin_datasets()

    def keys(self):
        return get_builtin_datasets().keys()

    def values(self):
        return get_builtin_datasets().values()

    def items(self):
        return get_builtin_datasets().items()

    def get(self, key, default=None):
        return get_builtin_datasets().get(key, default)


BUILTIN_DATASETS = _BuiltinDatasetsProxy()


def download_builtin_dataset(name):

    datasets = get_builtin_datasets()
    dataset = datasets[name]

    print("Trying to download dataset from " + dataset.url + "...")
    tmp_file_path = join(get_dataset_dir(), "tmp.zip")

    # Try to download with normal SSL verification first
    try:
        urlretrieve(dataset.url, tmp_file_path)
    except URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("SSL certificate verification failed. "
                  "Trying with unverified context...")
            # Create an unverified SSL context as fallback
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Use urlopen with custom SSL context and manually save the file
            try:
                with urlopen(dataset.url, context=ssl_context) as response:
                    with open(tmp_file_path, 'wb') as f:
                        f.write(response.read())
            except Exception as fallback_error:
                print(f"Failed to download dataset: {fallback_error}")
                raise
        else:
            # Re-raise other URLErrors
            raise

    with zipfile.ZipFile(tmp_file_path, "r") as tmp_zip:
        tmp_zip.extractall(join(get_dataset_dir(), name))

    os.remove(tmp_file_path)
    print("Done! Dataset", name, "has been saved to", join(get_dataset_dir(), name))
