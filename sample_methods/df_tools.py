import pandas as pd
import os 
import urllib.request

from typing import Union

def observations_to_df(observations_file: str, multimedia_file: str, delim: str = '\t', save_to: str = None) -> pd.core.frame.DataFrame:

  occurence_table = pd.read_csv(observations_file, delimiter=delim)
  media = pd.read_csv(multimedia_file, delimiter=delim)
  taxa = occurence_table[["gbifID","family","genus","species"]]
  images = media[["gbifID","identifier"]]
  data = pd.merge(images,taxa,on="gbifID", how="inner")
  data = data.dropna()

  if save_to != None:
    data.to_csv(save_to)

  return data

def get_taxa_freq(taxa_data_frame: Union[pd.core.frame.DataFrame, str]):
  
  if type(taxa_data_frame == str):
    taxa_data_frame = pd.read_csv(taxa_data_frame)

  families_freq = taxa_data_frame['family'].value_counts(sort=True)
  genera_freq = taxa_data_frame['genus'].value_counts(sort=True)
  species_freq = taxa_data_frame['species'].value_counts(sort=True)
  families_freq.to_csv('./freq_tables/families_freq.csv')
  genera_freq.to_csv('./freq_tables/genera_freq.csv')
  species_freq.to_csv('./freq_tables/species_freq.csv')

def sample(df: pd.core.frame.DataFrame, ut: int, lt: int, label_col: str, labels: list, label_freq: list) -> pd.core.frame.DataFrame:

  if label_freq[0] > ut:
    sample = df[df[label_col] == labels[0]].sample(ut, replace = False)
  elif label_freq[0] > lt:
    sample = df[df[label_col] == labels[0]]
  else:
    sample = pd.DataFrame({'gbifID' : []})

  for i in range(1,len(labels)):

    if label_freq[i] > lt:

      if label_freq[i] > ut:
        label_sample = df[df[label_col] == labels[i]].sample(ut, replace = False)
            
      else:
        label_sample = df[df[label_col] == labels[i]]

      sample = pd.concat([sample, label_sample], axis=0)

  return sample


