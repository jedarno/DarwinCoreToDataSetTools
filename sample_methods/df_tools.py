import pandas as pd
import os 
import urllib.request

def observations_to_df(observations_file: str, multimedia_file: str, delim: str = '\t', save_to: str = None) -> pd.core.frame.DataFrame:
  occurence_table = pd.read_csv(observations_file, delimiter=delim)
  media = pd.read_csv(multimedia_file, delimiter=delim)
  taxa = occurence_table[["gbifID","family","genus","species"]]
  images = media[["gbifID","identifier"]]
  data = pd.merge(images,taxa,on="gbifID", how="inner")
  data = data.dropna()
  if save_to != None:
    try:
      data.to_csv(save_to)
  return data


