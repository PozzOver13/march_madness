import os

from ncaa.src.loader import Loader

from ncaa.src.config.general import PATH_DATA_RAW


def test_load_data():
    loader = Loader()
    path_file = os.path.join(PATH_DATA_RAW, 'kaggle', 'cbb.csv')
    df = loader.load_data(path_file)

    assert df.shape[0] == 3523