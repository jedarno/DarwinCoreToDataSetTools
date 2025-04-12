import cv2
import numpy as np
import os
import requests

def build_image_dir(labels: list, folder_location: str = "images"):
  """
  Builds the file structure for storing the downloaded images. If no directory is provided a folder called images will be made.
  """

  if not os.path.exists(folder_location + "/train/"):
   os.makedirs(folder_location + "/train/")

  if not os.path.exists(folder_location + "/test/"):
   os.makedirs(folder_location + "/test/")

  if not os.path.exists(folder_location + "val/"):
    os.makedirs(folder_location + "/val/")

  for label in labels:
    if not os.path.exists(folder_location + "/train/" + label):
      os.makedirs(folder_location + "/train/" + label)

    if not os.path.exists(folder_location + "/test/" + label):
      os.makedirs(folder_location + "/test/" + label)

    if not os.path.exists(folder_location + "/val/" + label):
      os.makedirs(folder_location + "/val/" + label)

def download_image(image_link: str, path: str):
  """
  Download image from a link. The image is resized to 1090 x 1600. 
  """
  img_data = requests.get(image_link, timeout=5).content

  with open(path, 'wb') as handler:
    handler.write(img_data)

  #resize image
  image = cv2.imread(path)
  image = cv2.resize(image, (1090, 1600))
  cv2.imwrite(path, image)

def _resample(replacement_df, label_id):
  alternative_images = replacement_df[ (replacement_df[label_id] == labels[i]) & (~replacement_df['gbifID'].isin(gbif_ids)) ]
      
  if len(alternative_images > 0):

    for j in range (0, len(alternative_images)):

      try:
        alt_image_name = "image-{}-{}".format(alternative_images.iloc[j]['gbifID'], j)
        alt_path = "train/{}/{}.png".format(class_folder, alt_image_name)
        download_image(alternative_images.iloc[j]['identifier'], alt_path)
        replaced = True
        return True

      except Exception as e:
        print("Exception {} at link {}".format(e, alternatve_images.iloc[j]['identifier']))
  
  return False


def download_sample(directory, sample_df, label_id, replacement_df = False):
  """
  Donwload image links from sample into the dataset directory
  """

  broken_links = []
  missing_images = []
  labels = sample_df[label_id].to_list()
  gbif_ids = sample_df['gbifID'].to_list()
  image_links = sample_df['identifier'].to_list()
  
  for i in range(0, len(labels)):
    class_folder = labels[i]
    image_name = "image-{}-{}".format(gbif_ids[i], i)
    path = "{}/train/{}/{}.png".format(directory, class_folder, image_name)

    try:
      download_image(image_links[i], path)
    except Exception as e:
      print("Exception {} at link {}".format(e, image_links.iloc[i]['identifier']))

      if (replacement_df):
        broken_links.append(i)
        replaced = _resample(replacement_df, label_id)

        if replaced == True:
          print("Replacement image found")
       

def split_test_images(directory, labels):
  
  for label in labels:
    train_path = "{}/train/{}".format(directory, label)
    test_path = "{}/test/{}".format(directory, label)

    try:
      images = [img for img in os.listdir(train_path)]
    except:
      print("Warning! folder not found at {}".format(train_path))

    num_samples = int(np.floor(len(images) * 0.1))

    if num_samples == 0:
      num_samples = 1

    test_images = np.random.choice(images, num_samples, replace=False)

    for image in test_images:
      
      try:
        os.replace("{}/train/{}/{}".format(directory, label, image),  "{}/test/{}/{}".format(directory, label, image))
      except:
        print("image move failed {}".format(image))

def split_val_images(directory, labels):

  for label in labels:
    train_path = "{}/train/{}".format(directory, label)
    val_path = "{}/val/{}".format(directory, label)

    try:
      images = [img for img in of.listdir(train_path)]
    except:
      print("Warning! folder not found at {}".format(train_path))

    num_samples = int(np.floor(len(images) * 0.2))

    if num_samples == 0:
      num_samples = 1

    val_images = np.random.choice(images, num_samples, replace=False)

    for image in val_images:

      try:
        os.replace("{}/train/{}/{}".format(directory, label, image), "{}/val/{}/{}".format(directory, label, image))
      except:
        print("image move failed {}".format(image))


