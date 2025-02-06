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

  for label in labels:
    if not os.path.exists(folder_location + "/train/" + label):
      os.makedirs(folder_location + "/train/" + label)
    if not os.path.exists(folder_location + "/test/" + label):
      os.makedirs(folder_location + "/test/" + label)

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

def download_sample(directory, sample_df, label_id, replacement_df):
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
    path = "train/{}/{}.png".format(class_folder, image_name)

    try:
      download_image(image_links[i], path)
    except Exception as e:
      broken_links.append(i)
      replaced = False
      alternative_images = replacement_df[ (replacement_df[label_id] == labels[i]) & (~replacement_df['gbifID'].isin(gbif_ids)) ]
      
      if len(alternative_imags > 0):

        for j in range (0, len(alternative_images)):

          try:
            alt_image_name = "image-{}-{}".format(alternative_images.iloc[j]['gbifID'], j)
            alt_path = "train/{}/{}.png".format(class_folder, alt_image_name)
            download_image(alternative_images.iloc[j]['identifier'], alt_path)
            replaced = True
            break

          except Exception as e:
            print("Exception {} at link {}".format(e, alternatve_images.iloc[j]['identifier']))

        if not replaced:
          missing_images.append(labels[i])

      else:
        missing_images.append(labels[i])




