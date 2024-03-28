import polars as pl

class Loader:
    def load_data(self, path):
        df = pl.read_csv(path, separator=',')

        return df