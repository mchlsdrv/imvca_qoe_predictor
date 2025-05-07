import os
import pathlib
import pandas as pd


RES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/experiments/chatgpt_results.json')
RES_FILE.is_file()

res_df = pd.read_json(RES_FILE)
res_df

res_df.loc[:, ['Feature Type', 'BRISQUE', 'PIQE', 'FPS']].groupby("Feature Type").agg('mean')
res_df.loc[:, ['Feature Type', 'BRISQUE', 'PIQE', 'FPS']].groupby("Feature Type").agg('std')
