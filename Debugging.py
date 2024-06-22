from eval import loadPrices as formatted
from UN_OWEN_WAS_HER import getMyPosition as getPosition
import numpy as np
import pandas as pd

pricesFile = "./prices.txt"
#The transposed file
prcAll = formatted(pricesFile)
print("=== DEBUGGING ===")
print(prcAll.shape)
print("prcSoFar[:, -1]: \n",  prcAll[:,-1])
#print("prcSoFar[:, -2]: \n",  prcAll[:,-2])
#print("prcSoFar[:, -1] / prcSoFar[:, -2]: \n",  prcAll[:,-1]/ prcAll[:, -2])
#print(np.zeroes(prcAll.shape))