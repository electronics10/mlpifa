from settings import BLOCKS_NUM, REGIONX, REGIONY
import pandas as pd
import os

os.makedirs("data/comparison", exist_ok=True)
filePath = 'data/comparison/post_prediction.csv'
df = pd.read_csv(filePath, header=None)

for case in range(len(df)):
    fx = df.iloc[case, 0]
    Lmax = REGIONX - fx - 0.1
    Dmax = fx - 0.1
    Hmax = REGIONY - 0.1
    if df.iloc[case, BLOCKS_NUM+1] > Lmax: df.iloc[case, BLOCKS_NUM+1]=Lmax
    if df.iloc[case, BLOCKS_NUM+2] > Dmax: df.iloc[case, BLOCKS_NUM+2]=Dmax
    if df.iloc[case, BLOCKS_NUM+3] > Hmax: df.iloc[case, BLOCKS_NUM+3]=Hmax

df.to_csv(filePath, index=False, header=None)
