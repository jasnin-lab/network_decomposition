import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import morphology
from scipy.stats import pearsonr
from src.pipeline.mask_processing import *
from src.pipeline.graph_constructor import *
from src.pipeline.synthetic_dataset_generation import *

if __name__ == '__main__':
    list = [1, 2, 3, 4]
    print(list*4)
    print([i*4 for i in list])


