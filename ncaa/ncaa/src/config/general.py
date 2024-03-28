import os

# Path to the root of the project
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Path to the data directory
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_DATA_RAW = os.path.join(PATH_DATA, 'raw')