'''
Functions for dealing with images in MicroDraw
'''

import numpy as np
import urllib
import json
import re
from skimage.io import imread
import subprocess as sp

def get_dataset_description(source):
  '''Get the JSON object describing the dataset at `source`

  NOTE: This should be integrated in microdraw.py/download_dataset_definition
  [web][io]
  '''

  data = urllib.request.urlopen(source)
  return json.loads(data.read())

def get_microdraw_slice_size(
  source,
  slice_index,
  dataset_description = None
):
  '''Get the width and height of the slice. Values are obtained
  from the dzi xml file if that is provided, or from the width
  and height properties of the dictionary if the tileSource is
  provided through a formula.
  [web][io]'''
  if dataset_description is None:
    dataset_description = get_dataset_description(source)

  dzi_url = dataset_description["tileSources"][slice_index]
  if isinstance(dzi_url, dict):
    W, H = dzi_url["width"], dzi_url["height"]
    return W, H
  elif urllib.parse.urlparse(dzi_url).scheme == '':
    # dzi_url is relative to microdraw
    base_url = urllib.parse.urlparse(source).netloc
    base_protocol = urllib.parse.urlparse(source).scheme
    url = base_protocol + "://" + base_url + dzi_url
  else:
    # dzi_url is absolute
    url = dzi_url

  data = urllib.request.urlopen(url)
  txt = data.read().decode('ascii')
  W = int(re.findall('Width="(\d+)"', txt)[0])
  H = int(re.findall('Height="(\d+)"', txt)[0])
  return W, H

def get_max_scale_levels(
  W,
  H
):
  '''Get the maximum scale level for a given image width and height'''
  return np.ceil(np.log2(np.max([W,H])))
    
def get_nrows_ncols(
  W, H, scale_level, tile_size=2**8
):
  '''Get the number of columns and rows at a given scale level'''
  max_levels = get_max_scale_levels(W, H)
  return (
    int(np.ceil(W/2**(max_levels-scale_level)/tile_size)),
    int(np.ceil(H/2**(max_levels-scale_level)/tile_size))
  )

def _get_dzi_image_format(
  dzi_url
):
  '''Returns the image format in which the dzi tiles are encoded
  [web]
  '''
  res = urllib.request.urlopen(dzi_url).read()
  ext = [r[1] for r in [row.split("=") for row in str(res).split(" ")] if len(r)==2 and r[0]=="Format"][0].replace('"','')
  ext = ext.replace("\\n", "")
  return ext.strip(" \t\n\r")


def _get_tile_from_function(
  tile_fn, l, row, col
):
  tile_fn = sp.check_output([
    '/usr/local/bin/node',
    '-e',
    'const fn=' + tile_fn + ';process.stdout.write(fn(%i,%i,%i))'%(l,row,col)
  ])
  return tile_fn.decode('utf8')

def get_slice_image_array_at_scale_level(
  source,
  slice_index,
  scale_level,
  dataset_description = None
):
  '''Download all images corresponding to a given scale level
  [web]
  '''

  if dataset_description is None:
    dataset_description = get_dataset_description(source)
  W, H = get_microdraw_slice_size(source, slice_index)
  # max_scale_levels = get_max_scale_levels(W, H)
  ncols, nrows = get_nrows_ncols(W, H, scale_level)

  dzi_url = dataset_description["tileSources"][slice_index]
  base_url = urllib.parse.urlparse(source).netloc
  base_protocol = urllib.parse.urlparse(source).scheme
  images = []
  if isinstance(dzi_url, dict):
    file_base_url = base_protocol + "://" + base_url
    for row in range(nrows):
      row_arr = []
      for col in range(ncols):
        tile_url_string = _get_tile_from_function(dzi_url["getTileUrl"], scale_level, col, row)
        url = file_base_url + tile_url_string
        image = imread(url)
        row_arr.append(image)
      images.append(row_arr)
  else:
    if urllib.parse.urlparse(dzi_url).scheme == '':
      # dzi_url is relative to microdraw
      dzi_url = base_protocol + "://" + base_url + dzi_url
    ext = _get_dzi_image_format(dzi_url)
    files_url = dzi_url.replace(".dzi","_files")
    for row in range(nrows):
      row_arr = []
      for col in range(ncols):
        url = files_url + "/%i/%i_%i.%s"%(scale_level,col,row, ext)
        image = imread(url)
        row_arr.append(image)
      images.append(row_arr)
  return  images

def _get_concatenated_image_size(
  images
):
  '''Get the size of a concatenated image'''
  shape = None
  row = 0
  for col,im in enumerate(images[row]):
    if col == 0:
      shape = np.array(im.shape)
    else:
      shape[1] += im.shape[1]
  for row in range(1,len(images)):
    shape[0] += images[row][0].shape[0]
  return shape

def get_concatenated_image(
  images
):
  '''Concatenate all images in the image array'''
  shape = _get_concatenated_image_size(images)
  slice_image = np.zeros(shape, dtype="int")
  offset_row = 0
  for row in range(len(images)):
    offset_col = 0
    for col in range(len(images[row])):
      im = images[row][col]
      dims = len(im.shape)
      if dims == 2:
        slice_image[
          offset_row:(offset_row+im.shape[0]),
          offset_col:(offset_col+im.shape[1])] = im
      else:
        slice_image[
          offset_row:(offset_row+im.shape[0]),
          offset_col:(offset_col+im.shape[1]),
          :] = im
      
      offset_col += im.shape[1]
    offset_row += im.shape[0]
  return slice_image

def get_slice_image(
  source, slice_index, scale_level
):
  '''Get the slice image for the given scale level
  [unreferenced]
  '''
  dataset_description = get_dataset_description(source)
  image_array = get_slice_image_array_at_scale_level(
    source,
    slice_index,
    scale_level,
    dataset_description=dataset_description)
  return get_concatenated_image(image_array)
