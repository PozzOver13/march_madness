from ncaa.src.loader import Loader

def test_load_data():
    loader = Loader()
    str_output = loader.load_data('data.csv')

    assert str_output == 'Loading data from data.csv'