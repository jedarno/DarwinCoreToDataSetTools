import os

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


