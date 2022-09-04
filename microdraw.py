# pylint: disable=invalid-name

"""microdraw.py: functions for working with MicroDraw"""

import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon, MultiPolygon, LinearRing
from shapely.ops import nearest_points
from skimage import io
from skimage.draw import polygon, polygon_perimeter
from skimage.filters import gabor
from svgpathtools import Path, CubicBezier, Line
from tqdm import tqdm
import hashlib
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sys
import torch
import urllib.request as urlreq
import urllib.parse as urlparse
from layers import SinkhornDistance
import pyclipper
import compute_inner_contours as cic
import contourOptimalTransportToolbox as cOTT
import contourToolbox as cT
import generateProfilesToolbox as gpT
import image
import mesh
import requests
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
from scipy.linalg import orthogonal_procrustes

def version():
  '''Code version
  [utilities][unreferenced]
  '''
  print("1")

def download_project_definition(
  project,
  token
):
  '''Download a project. A project can contain several datasets

  [web][io][unreferenced]
  '''

  url = f"https://microdraw.pasteur.fr/project/json/{project}?token={token}"
  with urlreq.urlopen(url) as res:
    txt = res.read()
  prj = json.loads(txt)

  return prj

def download_dataset_definition(
  source
):
  '''Download a dataset. A dataset can contain several slices, each with
  several regions
  [web][io]
  '''

  with urlreq.urlopen(source) as res:
    txt = res.read()
  prj = json.loads(txt)

  return {
    "pixelsPerMeter": prj["pixelsPerMeter"],
    "numSlices": len(prj["tileSources"]),
    "project": prj
  }

def download_all_regions_from_dataset_slice(
  source, project, slce, token,
  backups=False,
  microdraw_url="https://microdraw.pasteur.fr"
):
  '''Download all regions in a dataset slice
  [web][io]
  '''

  url = f"{microdraw_url}/api?source={source}&project={project}&slice={slce}&token={token}"

  if backups is True:
    url = url + "&backup"

  with urlreq.urlopen(url) as res:
    txt = res.read()

  return json.loads(txt)

def download_all_regions_from_dataset_slice_array(
  source,
  project,
  token,
  slice_array,
  microdraw_url="https://microdraw.pasteur.fr"
):
  '''Download all regions in a an array of dataset slices

  [web][io][unreferenced]'''

  dataset = download_dataset_definition(source)
  dataset["slices"] = [[] for _ in range(dataset["numSlices"])]
  for i in slice_array:
    dataset["slices"][i] = download_all_regions_from_dataset_slice(
      source, project, i, token,
      microdraw_url=microdraw_url)
  return dataset

def download_all_regions_from_dataset(
  source,
  project,
  token,
  microdraw_url="https://microdraw.pasteur.fr"
):
  '''Download all regions in all slices in a dataset

  [web][io][unreferenced]'''

  dataset = download_dataset_definition(source)
  dataset["slices"] = []
  for i in tqdm(range(dataset["numSlices"])):
    dataset["slices"].append(download_all_regions_from_dataset_slice(
      source, project, i, token,
      microdraw_url=microdraw_url))
  return dataset

def _paperjs_path_to_resampled_polygon(co):
  '''
  Convert Paper.js paths to a resampled polygon (a line composed of
  consecutive vertices).
  Used by `download_microdraw_contours_as_polygons`.
  
  Note: Paper.js Bézier curves are encoded as px, py, ax, ay, bx, by,
  where px, py is an anchor point, ax, ay is the previous handle and
  bx, by the next handle. SVG path tools use a more standard encoding:
  p1x,p1y, b1x, b1y, a2x, a2y, p2x, p2y, where p1x, p1y is the start
  anchor point, p2x, p2y the end anchor point, b1x, b1y is the
  handle coming out from p1x, p1y, and a2x, a2y is the handle entering
  into the end anchor point.
  [paperjs][conversion]
  '''
  mysegs = []
  for i in range(len(co)):
    c = co[i%len(co)]
    c1 = co[(i+1)%len(co)]
    try:
      if isinstance(c[0], (list, np.ndarray)):
        # print("c[0] is list")
        [s,sj],[_,_],[b,bj] = c
      else:
        # print("c[0] is not list:", type(c[0]))
        s,sj,b,bj = c[0],c[1],0,0
      if isinstance(c1[0], (list, np.ndarray)):
        # print("c1[0] is list")
        [s1,s1j],[a1,a1j],[_,_] = c1
      else:
        # print("c1[0] is not list:", type(c1[0]))
        s1,s1j,a1,a1j = c1[0],c1[1],0,0
      # print("c:", c)
      # print("c1:", c1)
      # print("s,sj:",s,sj,", b,bj:",b,bj, ", s1,sij:", s1,s1j, ", a1,a1j:",a1,a1j)
      seg = CubicBezier(
        complex(s,sj),
        complex(s+b,sj+bj),
        complex(s1+a1,s1j+a1j),
        complex(s1,s1j))
      mysegs.append(seg)
    except: # ValueError as err:
        # print(err)
        pass
  if len(mysegs) <5:
      # print("len(mysegs) is < 5")
      return
  p = Path(*mysegs)
  NUM_SAMPLES = int(p.length())
  my_path = []
  for i in range(NUM_SAMPLES):
      x = p.point(i/(NUM_SAMPLES-1))
      my_path.append([x.real,x.imag])
  return np.array(my_path)

def convert_microdraw_contours_to_resampled_polygons(
  conts,
  width,
  included_regions=None
):
  '''Convert MicroDraw contours to polygons. MicroDraw contours
  are Paper.js polygons or Bézier curves. Polygons are arrays
  of 2D coordinates.

  For a similar functionality, but without resampling, see
  `get_regions_from_dataset_slice`.

  Parameters
  ----------
  conts: MicroDraw contours
    A MicroDraw contour, which are Paper.js contours.
  width: Width of the slice image
    Width of the slice image where the contour was drawn.
  included_regions: List of region names to include.
    If provided, only regions whose name is in the list will be included.

  Returns
  -------
  polys: A list of polygons corresponding to the individual
    contours in `conts`. Each polygon is an array of 2D
    coordinates.
  [paperjs][conversion]
  '''

  polys = []
  for i in range(len(conts)):
    if included_regions is not None:
      if conts[i]["annotation"]["name"] not in included_regions:
        continue

    root = conts[i]["annotation"]["path"][1]

    if "segments" in root:
      poly = _paperjs_path_to_resampled_polygon(np.array(root["segments"]))
      if poly is not None:
          poly = poly*width/1000 # scale to match image size
          polys.append(poly)
      # else:
      #     print("segments poly is None")
      #     print(poly)
    if "children" in root:
      for cont in root["children"]:
        poly = _paperjs_path_to_resampled_polygon(np.array(cont[1]["segments"]))
        if poly is not None:
          poly = poly*width/1000 # scale to match image size
          polys.append(poly)
        # else:
        #   print("children poly is none")
  return polys

def download_microdraw_contours_as_polygons(
  source,
  project,
  slice_index,
  token,
  width
):
  '''[io][web][conversion]'''
  conts = download_all_regions_from_dataset_slice(
    source, project, slice_index, token)
  if len(conts) == 0:
    # print("len(conts) is 0")
    return None
  polys = convert_microdraw_contours_to_resampled_polygons(
    conts, width)
  return polys

def upload_polygons_to_microdraw_slice(
  source,
  project,
  slice_index,
  token,
  polygons,
  name,
  rgb,
  host="https://microdraw.pasteur.fr/api/upload"
):
    '''Upload polygons to a slice in a MicroDraw project. Polygon coordinates
    should be in svg space.

    Parameters
    ----------

    polygons: list of lists
      List of polygons, each polygon is a list of 2D vertices (and not a numpy array!)

    [web][io][unreferenced]
    '''

    text = convert_polygons_to_microdraw_json(polygons, name, rgb)

    payload = {
      "action": "save",
      "source": source,
      "slice": slice_index,
      "project": project,
      "token": token
    }

    resp =  requests.post(host, files = {"data": ("polygon.json", text)}, params=payload)

    return resp

def get_points_from_segment(
  seg
):
  '''Get points from segment
  [paperjs]
  '''

  points = []
  for i in range(seg.shape[0]):
    seg[i] = np.array(seg[i])
    if type(seg[i][0]).__name__ == "ndarray":
      # flatten curve segments
      points.append([seg[i][0][0], seg[i][0][1]])
    else:
      # line segment
      points.append([seg[i][0], seg[i][1]])
  return points

def get_regions_from_dataset_slice(
  dataset
):
  '''Get regions from dataset slice
  [io][paperjs]'''

  regions = []
  for region in dataset:
    name = region['annotation']['name']
    path_type = region['annotation']['path'][0]
    if path_type == "Path" and 'segments' in region['annotation']['path'][1]:
      seg = np.asfortranarray(region['annotation']['path'][1]['segments'])
      points = get_points_from_segment(seg)
      regions.append((name, np.array(points), 0))
    elif path_type == "CompoundPath":
      children = region['annotation']['path'][1]['children']
      for children_number,child in enumerate(children):
        if 'segments' in child[1]:
          segments = [np.asfortranarray(child[1]['segments'])]
          for seg in segments:
            points = get_points_from_segment(seg)
            regions.append((name, np.array(points), children_number))
  return regions

def find_compound_regions(
  regions,
  min_points_in_region=3
):
  '''Find compound regions: RENAMED'''
  raise ValueError("Renamed as find_compound_region_indices")

def find_compound_region_indices(
  regions,
  min_points_in_region=3
):
  '''Create an array of non-garbage compound region indices. The regions are
  all in the same slice. Regions in microdraw can be garbage regions (very few
  points). They can also be compound, and contain subregions. These subregions
  can also be garbage. This function finds all non-garbage regions and builds
  an array of indices for all their non-garbage subregions. The index of the
  main array is the same as the index of the original regions. Within each of
  these entries, the indices correspond to the index of the subregion within
  the region.

  [utilities][unreferenced]
  '''

  compound_region_indices = []
  i=0
  while i < len(regions):
    _, region, _ = regions[i]

    # skip potential garbage at the begining
    while len(region) < min_points_in_region and i < len(regions)-1:
      i += 1
      _, region, _ = regions[i]

    # break if reached the end
    if i > len(regions)-1 or len(region) < min_points_in_region:
      break

    # add first non-garbage region and eventual children
    sub_regions = [i]
    while i < len(regions)-1:
      if regions[i + 1][2] == 0:
        break
      i += 1
      if len(regions[i][1]) >= min_points_in_region:
        sub_regions.append(i)
    compound_region_indices.append(sub_regions)
    i += 1
  return compound_region_indices

def is_reference_orientation_a_hole(
  regions,
  sub_regions
):
  '''Determine the orientation of sub regions in a compound region
  [shapely]
  '''
  reference_orientation_is_hole = False
  if len(sub_regions) > 1:
    ref_region = regions[sub_regions[0]][1]
    poly_ref = Polygon(ref_region)
    for sub_region in sub_regions:
      region = regions[sub_region][1]
      poly_sub_region = Polygon(region)

      if not poly_ref.intersects(poly_sub_region):
        continue

      area = Polygon(poly_ref.exterior.coords, [poly_sub_region.exterior.coords]).area
      if area < 0:
        reference_orientation_is_hole = True
        break
  else:
    region = regions[sub_regions[0]][1]
    reference_orientation_is_hole = False
  return reference_orientation_is_hole

def normalise_compound_region(
  compound_region_indices,
  regions
):
  '''Normalise a compound region so that exterior contours are
  counterclockwise and holes are clockwise.
  [shapely][unreferenced]
  '''
  _, sub_regions = compound_region_indices

  reference_orientation = LinearRing(regions[sub_regions[0]][1]).is_ccw
  reference_orientation_is_hole = is_reference_orientation_a_hole(regions, sub_regions)

  normalised_compound_region = []
  for sub_region in sub_regions:
    name, region, child_number = regions[sub_region]
    oriented_as_reference = LinearRing(region).is_ccw==reference_orientation
    is_hole = (oriented_as_reference
      if reference_orientation_is_hole
      else (not oriented_as_reference))

    normalised_region = region.copy()
    if is_hole and LinearRing(region).is_ccw:
      normalised_region = np.flip(region, axis=0)
    if not is_hole and not LinearRing(region).is_ccw:
      normalised_region = np.flip(region, axis=0)
    normalised_compound_region.append([name, normalised_region, child_number])
  return normalised_compound_region

def compound_region_to_polygon(
  region,
  compound_region_indices
):
  '''Deprecated: use compound_region_to_polygons instead.'''
  raise ValueError("Replaced by compound_region_to_polygons, which returns an array of polygons")

def compound_region_to_polygons(
  region,
  compound_region_indices
):
  '''Convert a compound region into an array of shapely polygons
  [conversion][shapely][unreferenced]'''
  poly = Polygon(
    region[compound_region_indices[0]][1],
    [region[sub_region_index][1] for sub_region_index in compound_region_indices[1:]])
  if poly.is_valid:
    return [poly]

  tmp = shapely.ops.polygonize_full(poly)
  polys = []
  for elem in tmp:
    for elem1 in elem.geoms:
      if type(elem1) is shapely.geometry.linestring.LineString:
        polys.append(Polygon(elem1))
      else:
        polys.append(elem1)
  return polys

def draw_all_dataset(
  dataset,
  ncol=13,
  width=800,
  alpha=0.5,
  path=None
):
  '''Draw all dataset
  [plotting][unreferenced]
  '''

  plt.figure(figsize=(25, 10))
  i = 0
  j = 0
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region, _ in regions:
      color = color_from_string(name)
      plt.fill(region[:, 0]+i*width, -region[:, 1]-j*width, alpha=alpha, c=color)
      plt.text((i+0.5)*width, -(j+1)*width, str(slce), alpha=0.5)
    i += 1
    if i >= ncol:
      i = 0
      j += 1
  plt.axis('equal')
  if path:
    plt.savefig(path)

def dataset_as_volume(
  dataset,
  filter=None
):
  '''Combine all regions into a single mesh. This mesh does not have
  triangles, only region contours.
  dataset: the dataset from which to obtain the region contours
  filter: an optional array of strings with the names of the regions
          to include in the mesh
  [conversion][mesh]
  '''

  verts = []
  eds = []
  neds = 0
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region, _ in regions:
      if filter != None and name not in filter:
        continue
      verts.extend([(x, y, slce) for x, y in region])
      eds.extend([(neds+i, neds+(i+1)%len(region)) for i in range(len(region))])
      neds = len(eds)
  return verts, eds

def save_dataset_as_text_mesh(
  dataset,
  path,
  voxdim=[0.1,0.1,1.25]
):
  '''Save dataset as a text mesh

  [conversion][io][unreferenced]'''

  verts, eds = dataset_as_volume(dataset)
  mesh_str = f"{len(verts)} 0 {len(eds)}"
  verts_str = "\n".join(
    f"{x * voxdim[0]} {y * voxdim[1]} {z * voxdim[2]}"
    for x, y, z in verts)
  eds_str = "\n".join(f"{i} {j}" for i, j in eds)
  mesh_str = "\n".join((mesh_str, verts_str, eds_str))

  with open(path, 'w') as file:
    file.write(mesh_str)

def dataset_to_nifti(
  dataset,
  voxdim=[0.1, 0.1, 1.25],
  region_name=None
):
  '''Convert dataset to nifti volume. Returns a nifti object
  [conversion]'''

  verts, _ = dataset_as_volume(dataset)
  vmin, vmax = np.min(verts, axis=0), np.max(verts, axis=0)
  vmin = np.floor(vmin)
  vmax = np.ceil(vmax)
  size = vmax-vmin
  size[2] = dataset["numSlices"]
  img = np.zeros([int(x) for x in size], 'uint8')
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region, _ in regions:
      if region_name is None or name in region_name:
        try:
          rows, cols = polygon(region[:, 0]-vmin[0], region[:, 1]-vmin[1], img.shape)
          img[rows, cols, slce] = 255
          rows, cols = polygon_perimeter(region[:, 0]-vmin[0], region[:, 1]-vmin[1], img.shape)
          img[rows, cols, slce] = 0
        except:
          continue
  affine = np.eye(4)
  affine[0, 0] = voxdim[0]
  affine[1, 1] = voxdim[1]
  affine[2, 2] = voxdim[2]
  return nib.Nifti1Image(img, affine=affine)

def save_dataset_as_nifti(
  dataset,
  path,
  voxdim=[0.1, 0.1, 1.25],
  region_name=None
):
  '''Save dataset as a nifti volume
  [conversion][io][unreferenced]
  '''

  nii = dataset_to_nifti(dataset, voxdim, region_name)
  nib.save(nii, path)

def load_dataset(
  path
):
  '''Load dataset stored in json format
  [io][unreferenced]
  '''

  dataset = None
  with open(path, 'r') as infile:
    dataset = json.load(infile)
  return dataset

def save_dataset(
  data,
  path
):
  '''Save dataset in json format
  [io][unreferenced]
  '''

  with open(path, 'w') as outfile:
    json.dump(data, outfile)

def color_from_string(
  my_string
):
  '''Create a random color based on a hash of the input string
  [utilities]
  '''

  sha = hashlib.sha256()
  sha.update(my_string.encode())
  return "#" + sha.hexdigest()[:6]

def save_contour(
  path,
  verts,
  contour
):
  '''A contour is a list of vertices and a list of edges. A contour can
  contain multiple independent closed contours, but the list of vertices
  and edges is always unique. This function saves the contour in a
  format readable by MeshSurgery.

  [conversion][io][mesh][unreferenced]
  '''
  with open(path, "w") as fi:
    fi.write(f"{len(verts)} 0 {len(contour)}\n")
    fi.write("\n".join(f"{cx} {cy} 0" for (cx,cy,_) in verts) + "\n")
    fi.write("\n".join(str(a)+" "+str(b) for (a,b) in contour))

def raw_contour(
  v,
  f,
  x
):
  raise ValueError("Moved to mic.mesh.raw_contour")

def no_duplicates_contour(
  vert_point_and_coord,
  contour
):
  raise ValueError("Moved to mic.mesh.no_duplicates_contour")

def continuous_contours(
  edge_soup
):
  raise ValueError("Moved to mic.mesh.continous_contours")

def slice_mesh(
  v,
  f,
  z,
  min_contour_length=10
):
  raise ValueError("Moved to mic.mesh.slice_mesh")

def scale_contours_to_image(
  v,
  width,
  height,
  scale_yz
):
  raise ValueError("Moved to mic.mesh.scale_contours_to_image")

def draw_slice(
  v,
  f,
  slice_index,
  path,
  scale_x=3,
  scale_yz=0.25,
  slice_offset=0
):
  '''[plotting][unreferenced]'''
  his = io.imread(path)
  iH, iW, _ = his.shape
  ve, _, _, _ = mesh.slice_mesh(v, f, (slice_index-slice_offset)*scale_x)
  ve = mesh.scale_contours_to_image(ve, iW, iH, scale_yz)
  plt.imshow(his)
  plt.scatter(ve[:,0],ve[:,1])

def save_slice_contours_in_svg(
  path,
  contours,
  width,
  height
):
  '''[conversion][io][unreferenced]
  '''
  poly = MultiPolygon([Polygon(c) for c in contours])
  svg_start = '''<svg
    xmlns="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    width="{width}"
    height="{height}"
    viewBox="0,0,{width},{height}"
  >
  '''.format(width=width,height=height)
  svg_content = poly.svg(scale_factor=2,fill_color="#000000")
  svg_end = '''
  </svg>'''
  svg = svg_start + svg_content + svg_end
  svg = svg.replace('opacity="0.6"', 'opacity="1"').replace('stroke="#555555"', 'stroke="#ffffff"')
  with open(path,"w") as file:
    file.write(svg)

def convert_polygons_to_microdraw_json(
  polygons,
  name,
  rgb
):
  '''Convert an array of polygons (a series of vertices) into Microdraw JSON
  format (which is the default Paper.js format)

  [conversion][unreferenced]
  '''
  annotation_text = '[\n'
  annotations = []
  for poly in polygons:
    annotation = '''{
      "annotation":{"path":["Path",{"applyMatrix":true,"segments":%s,"closed":true,"fillColor":[%s],"strokeColor":[0,0,0],"strokeScaling":false}
      ],"name":"%s"}}'''%(poly, rgb, name)
    annotations.append(annotation)
  annotation_text += ",\n".join(annotations)
  annotation_text += ']'
  return annotation_text

def icp_step(
  ref,
  mov
):
  '''One step of the iterative closest point method'''
  # out = [
  #   nearest_points(Polygon(ref), Point(pt))[0].coords[0]
  #   for pt in mov]
  M = distance_matrix(ref, mov)
  closest_ref = np.argmin(M, axis=0)
  out = ref[closest_ref]

  out = np.array(out)
  oout = Polygon(out).centroid.coords[0]
  omov = Polygon(mov).centroid.coords[0]
  C = np.zeros([2,2])
  for i in range(len(mov)):
    V1 = np.array([mov[i]-omov])
    V2 = np.array([out[i]-oout])
    C = C + V1.T*V2
  U, _, V = np.linalg.svd(C)
  R = U.dot(V)
  t = omov - R.dot(oout)
  mov = mov.dot(R)
  mov = mov - t
  dist = np.arccos((np.trace(R)-1)/2.0)
  dist = dist + np.linalg.norm(t)
  return mov, dist

def icp(
  ref,
  mov
):
  '''Iterative closest point method to align two polygons

  Parameters
  ----------
  ref: Array of shape (, 2)
    Reference polygon
  mov: Array of shape (, 2)
    Moving polygon

  Returns
  -------
  mov: Array of shape (, 2)
    A version of mov registered to ref
  diff: float
    A mesurement of the remaining error
  i: int
    Number of iterations
  [registration]
  '''

  maxiter=100
  tol=1e-6
  dist0=-1
  for i in range(maxiter):
    mov, dist = icp_step(ref, mov)
    if dist0<0:
      diff=dist
      print("initial distance:", diff)
    else:
      diff=np.abs(dist-dist0)
    if diff<tol:
      break
    dist0=dist
  print("final distance:", diff, "(after", i, "iterations)")
  return mov, diff, i

def icp_contours_step(
  ref_contours,
  mov_contours,
  max_dist=100
):
  '''One step of the iterative closest point method'''

  mov_pts = []
  for poly in mov_contours:
    mov_pts.extend(poly.tolist())
  mov_pts = np.array(mov_pts)

  ref_pts = []
  for poly in ref_contours:
    ref_pts.extend(poly.tolist())
  ref_pts = np.array(ref_pts)
  
  M = distance_matrix(ref_pts, mov_pts)
  closest_ref = np.argmin(M, axis=0)
  out = ref_pts[closest_ref]
  
  # use only points closer than max_dist
  dist = np.sum((out - mov_pts)**2, axis=1)**0.5
  selection = dist < max_dist
  out_sel = out[selection]
  mov_pts_sel = mov_pts[selection]

  if len(out_sel) < 10 or len(mov_pts_sel) < 10:
    return mov_contours, np.Infinity

  oout = np.mean(out_sel, axis=0)
  omov = np.mean(mov_pts_sel, axis=0)

  V1 = mov_pts_sel - omov
  V2 = out_sel - oout
  R, _ = orthogonal_procrustes(V1, V2)

  # C = np.zeros([2,2])
  # for i in range(len(mov_pts_sel)):
  #   V1 = mov_pts_sel[i] - omov
  #   V2 = out_sel[i] - oout
  #   C = C + V1.T*V2
  # U,S,V = np.linalg.svd(C)
  # R = U.dot(V)
  t = omov - R.dot(oout)

  # apply transformation
  for contour_index, contour in enumerate(mov_contours):
    contour = contour.dot(R)
    contour = contour - t
    mov_contours[contour_index] = contour

  dist = np.arccos((np.trace(R)-1)/2.0)
  dist = dist + np.linalg.norm(t)

  return mov_contours, dist

def icp_contours(
  ref_contours,
  mov_contours
):
  '''Iterative closest point method to align two shapely multipolygons

  Parameters
  ----------
  ref: An instance of shapely.geometry.MultiPolygon
    Reference multipolygon
  mov: An instance of shapely.geometry.MultiPolygon
    Moving multipolygon

  Returns
  -------
  mov: An instance of shapely.geometry.MultiPolygon
    A version of mov registered to ref
  diff: float
    A mesurement of the remaining error
  i: int
    Number of iterations
  [registration]
  '''

  maxiter=100
  tol=1e-6
  dist0=-1
  for i in range(maxiter):
    mov_contours, dist=icp_contours_step(ref_contours, mov_contours)
    if dist0<0:
      diff=dist
      print("initial distance:", diff)
    else:
      diff=np.abs(dist-dist0)
    if diff<tol:
      break
    dist0=dist
  print("final distance:", diff, "(after", i, "iterations)")

  if np.isnan(diff):
    return None, None, None
  return mov_contours, diff, i

def find_contour_correspondences(
  manual,
  auto,
  min_pct=0.7,
  max_pct=1.4
):
  '''Find correspondences between manually drawn contours in manual and
  mesh slice contours in auto. Contours are considered to correspond if
  their overlap is at least min_pct and at most max_pct.
  
  Parameters
  ----------
  manual: list of N arrays of shape (, 2)
    Manually drawn contours
  auto: list of M arrays of shape (, 2)
    Mesh slice contours
  min_pct: float
    Minimum fraction overlap
  max_pct: float
    Maximum fraction overlap
  
  Returns
  -------
  corresp: Matrix of shape (N, M)
    A matrix with rows corresponding to each of the manual contours
    and columns to each of the auto contours. Matrix entries are -1 if
    the pair of contours does not match, and indicates fraction overlap if
    they do match
  [registration]
  '''
  malen = len(manual)
  aulen = len(auto)
  m = np.ones((malen, aulen))*(-1)

  if malen>5 and aulen>5:
    for i in range(malen):
      for j in range(aulen):
        pola = Polygon(manual[i]).convex_hull
        polb = Polygon(auto[j]).convex_hull
        areaa = pola.area
        areab = polb.area
        areaaandb = pola.intersection(polb).area
        areaaorb = pola.union(polb).area

        overlap = areaaandb/areaaorb
        if overlap < min_pct:
          overlap = -1
        m[i,j] = overlap

  corresp = np.argmax(m.T,axis=0)
  for i,val in enumerate(corresp):
    if m[i,val] == -1:
      corresp[i] = -1
  return corresp

def register_corresponding_contour_pairs(
  manual,
  auto,
  corresp
):
  '''Register components contours using the iterative closest point method.
  Components of the manual and auto contours are registered in paris, according
  to the assignments in corresp.
  
  Parameters
  ----------
  manual: list of N arrays of shape (, 2)
    Manually drawn contours
  auto: list of M arrays of shape (, 2)
    Mesh slice contours
  corresp: array of shape (N, M)
    Matrix indicating the correspondence between individual contours
    in manual and auto, as provided by the `find_contour_correspondences`
    function
  
  Returns
  -------
  registered: list of arrays of shape (, 2)
    A version of the contours in manual registered to the corresponding
    contours in auto. `None` is returned for manual contours without
    correspondence among the auto contours.
  [registration]
  '''
  # print("corresp", corresp, corresp.shape)
  # print("malen, aulen:", len(manual), len(auto))
  registered = []
  for imanual, iauto in enumerate(corresp):
    print(f"Registering manual contour {imanual} to automatic contour {iauto}")
    mov = manual[imanual]
    if iauto < 0:
      print(f"Manual contour {imanual} does not have a corresponding automatic contour")
      registered.append(None) # old behaviour: registered.append(mov)
      continue
    ref = auto[iauto]
    mov_result, _, _ = icp(ref, mov)
    registered.append(mov_result)
  return registered

def register_manual_to_mesh_contour_pairs(
  manual,
  v,
  f,
  width,
  height,
  scale_x,
  scale_yz,
  slice_index,
  slice_offset
):
  '''Register manual contours for a slice with those from a
  corresponding mesh slice
  
    
  Parameters
  ----------
  manual: list of arrays of shape (, 2)
    MicroDraw contours as polygons in image coordinates
  v: array of floats with shape (, 3)
    Mesh vertices in mesh space (voxdim takes it to svg space)
  f: array of ints with shape (, 3)
    Mesh triangular faces
  scale_x: float
    voxdim[2]
  scale_yz: float
    voxdim[0] or voxdim[1]
  slice_index: int
    Slice index
  slice_offset: float
    (displacement[2]) slice where the mesh begins
  
  Returns
  -------
  registered: list of arrays of shape (, 2)
    A version of the contours in auto registered to the corresponding
    contours in manual.
  auto: list of arrays of shape (, 2)
    The original auto contours, reordered as in manual
  autoco: list of arrays of shape (, 3)
    Mesh-relative coordinates of the vertices of the contours in auto. The
    1st value is the index of the triangle in the mesh. The 2nd value is the
    index of the edge in the triangle (0, 1, 2). The 3rd value is the position
    p in the edge with p in [0, 1] (see mic.mesh.slice_mesh for more
    information)

  [registration]
  '''
  if manual is None:
    # print("manual is None")
    return None, None, None
  ve, veco, _, lines = mesh.slice_mesh(v, f, (slice_index - slice_offset) * scale_x)
  if ve is None:
    return None, None, None
  ve = mesh.scale_contours_to_image(ve, width, height, scale_yz)
  auto = [ve[line] for line in lines]
  autoco = []
  for line in lines:
    one_autoco = []
    for i in line:
      one_autoco.append(veco[i])
    autoco.append(one_autoco)
  if len(manual) == 0 or len(auto) == 0:
    # print("either len(manual) or len(auto) are 0:", len(manual), len(auto))
    return None, None, None

  corresp = find_contour_correspondences(manual, auto)

  if sum(corresp==-1) == len(corresp):
    print("No corresponding contours found")
    return None, auto, autoco

  registered = register_corresponding_contour_pairs(manual, auto, corresp)
  auto2 = []
  autoco2 = []
  for i in range(len(corresp)):
    if corresp[i] >= 0:
      auto2.append(auto[int(corresp[i])])
      autoco2.append(autoco[int(corresp[i])])
  return registered, auto2, autoco2

def register_contours(
  ref_contours,
  mov_contours
 ):

  '''If registration is not possible, the original data is returned'''

  # convert contours to multipolygon
  reg_contours, _, _ = icp_contours(ref_contours, mov_contours)

  if reg_contours is None:
    print("WARNING: Registration was not possible")
    return mov_contours

  return reg_contours

def register_manual_to_mesh_contours(
  manual,
  v,
  f,
  width,
  height,
  scale_x,
  scale_yz,
  slice_index,
  slice_offset
):
  '''Register manual contours for a slice with those from a
  corresponding mesh slice. The number of contours in each
  dataset does not need to be the same. Only contours close
  enough are used for the registration, but the resulting
  transformation is applied to all contours.
  
    
  Parameters
  ----------
  manual: list of arrays of shape (, 2)
    MicroDraw contours as polygons in image coordinates
  v: array of floats with shape (, 3)
    Mesh vertices in mesh space (voxdim takes it to svg space)
  f: array of ints with shape (, 3)
    Mesh triangular faces
  scale_x: float
    voxdim[2]
  scale_yz: float
    voxdim[0] or voxdim[1]
  slice_index: int
    Slice index
  slice_offset: float
    (displacement[2]) slice where the mesh begins
  
  Returns
  -------
  registered: list of arrays of shape (, 2)
    A version of the contours in auto registered to the contours in manual.
  mesh_contours: list of arrays of shape (, 2)
    Mesh contours for the slice
  mesh_contours_coords: list of arrays of shape (, 3)
    Mesh-relative coordinates of the vertices of the contours in mesh_contours.
    The 1st value is the index of the triangle in the mesh. The 2nd value is
    the index of the edge in the triangle (0, 1, 2). The 3rd value is the
    position p in the edge with p in [0, 1] (see mic.mesh.slice_mesh for more
    information)

  [registration]
  '''
  if manual is None:
    # print("manual is None")
    return None, None, None
  ve, veco, _, lines = mesh.slice_mesh(v, f, (slice_index - slice_offset) * scale_x)
  if ve is None:
    return None, None, None
  ve = mesh.scale_contours_to_image(ve, width, height, scale_yz)
  mesh_contours = [ve[line] for line in lines]
  mesh_contours_coords = []
  for line in lines:
    contour_coords = []
    for i in line:
      contour_coords.append(veco[i])
    mesh_contours_coords.append(contour_coords)
  if len(manual) == 0 or len(mesh_contours) == 0:
    # print("either len(manual) or len(auto) are 0:", len(manual), len(auto))
    return None, None, None

  registered = register_contours(mesh_contours, manual)

  return registered, mesh_contours, mesh_contours_coords

def plot_manual_contours_to_mesh_contours_result(
  manual,
  auto,
  registered
):
  if manual is not None and len(manual)>0:
    for poly in manual:
      plt.plot(poly[:, 0], poly[:, 1], 'r,', lw=0.5, label="manual")

  if auto is not None and len(auto)>0:
    for poly in auto:
      plt.plot(poly[:, 0], poly[:, 1], 'g.-', lw=0.5, label="mesh")

  if registered is not None and len(registered)>0:
    for poly in registered:
      if poly is not None and len(poly)>0:
        plt.plot(poly[:, 0], poly[:, 1], 'b,', lw=0.5, label="registered")

  plt.axis('equal')
  plt.legend()

def register_microdraw_contours_to_mesh_contours_for_slice(
  source,
  project,
  slice_index,
  token,
  v,
  f,
  width,
  height,
  scale_x,
  scale_yz,
  slice_offset
):
  '''Download slice contours from microdraw and register them with
  the contours from a corresponding mesh slice
  
  NOTE: This funtion should be removed.

  [registration][unreferenced]
  '''
  manual = download_microdraw_contours_as_polygons(
    source, project, slice_index,
    token, width)
  registered, auto, autoco = register_manual_contours_to_mesh_contours_for_slice(
    manual, v, f, width, height,
    scale_x, scale_yz,
    slice_index, slice_offset)
  return registered, manual, auto, autoco

def find_mesh_contour_to_profile_contour_correspondence(
    mesh_contour, # the contour obtained by slicing the mesh:
      # its points refer back to mesh vertices
    registered_manual_contour, # the manually drawn contour from microdraw,
      # registered to the mesh
    profile_contour # the manually drawn contour from microdraw,
      # non-registered, downsampled, used to extract profiles
):
  '''Align registered_manual_contour to mesh_contour the result is ind: a list
  giving for each point in registered_manual_contour the index of the closest
  point in mesh_contour
  
  NOTE: There's a function find_mesh_contour_to_profile_contour_correspondence
  in the script histology_to_mesh.py. It looks like that one is more accurate.
  Check it and replace it if that's the case!

  [registration][unreferenced]
  '''

  nsamples_mesh = len(mesh_contour)
  nsamples_regm = len(registered_manual_contour)
  nsamples_prof = len(profile_contour)

  regm_ring = LinearRing(registered_manual_contour)
  regm_contour = np.zeros((nsamples_mesh, 2))
  for i in range(nsamples_mesh):
    regm_contour[i,:] = regm_ring.interpolate(i/nsamples_mesh, normalized=True).coords
  scale=10000
  x = torch.tensor(mesh_contour/scale, dtype=torch.float)
  y = torch.tensor(regm_contour/scale, dtype=torch.float)
  sinkhorn = SinkhornDistance(eps=1e-5, max_iter=100, reduction=None)
  _, P, _ = sinkhorn(x, y)
  ind = assignments_indices(x, y, P)

  regm2reg = np.zeros(nsamples_mesh,dtype=int) # correspondence between
    # resample registered and raw registered
  reg2prof = np.zeros(nsamples_regm,dtype=int) # correspondence between
    # raw registered and profile_contour

  for j in range(nsamples_regm):
    le = regm_ring.project(
      Point(registered_manual_contour[j,:]),
      normalized=True)
    i = int(le*nsamples_mesh)%nsamples_mesh
    regm2reg[i] = j
    i = int(le*nsamples_prof)%nsamples_prof
    reg2prof[j] = i

  corresp = np.zeros(nsamples_mesh, dtype=int)
  for i in range(nsamples_mesh):
    i1=ind[i]
    i2=regm2reg[i1]
    i3=reg2prof[i2]
    corresp[i] = i3
  return corresp

def compute_inner_contours_for_polygons(
  polygons,
  iterations=10,
  projection="maxcoupling",
  scale=1000,
  nodeinterval=10,
  dist_cont=-3
):
  '''[inner-contours][unreferenced]
  '''
  return cic.compute_inner_contours_for_polygons(
    polygons,
    iterations=iterations,
    projection=projection,
    scale=scale,
    nodeinterval=nodeinterval,
    dist_cont=dist_cont)

def compute_inner_contour(
  coords_arr,
  delta
):
  '''[inner-contours][unreferenced]
  '''
  pco = pyclipper.PyclipperOffset()
  pco.AddPath(coords_arr, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
  ## compute "sub-curves"
  res = pco.Execute(delta)
  largest = 0
  index = 0
  for i,r in enumerate(res):
    if len(r)>largest:
      largest = len(r)
      index = i
  subcontour = np.asarray(pco.Execute(delta)[index])
  return subcontour

def _compute_inner_polygon_iterative(
  poly,
  iterations,
  dist_cont
):
  '''Compute inner polygon in steps
  [inner-contours]
  '''
  inner_polygon = poly
  for _ in range(iterations):
    inner_polygon = cOTT.computeInnerContour(
      inner_polygon,
      dist_cont)
  return inner_polygon

def _resample_outer_and_inner_polygons(
  outer_polygon,
  inner_polygon,
  nodeinterval
):
  '''Resample inner and outer polygons
  outer_polygon: array of shape (, 2)
  inner_polygon: array of shape (, 2)
  [inner-contours]
  '''
  outer = LinearRing(outer_polygon)
  inner = LinearRing(inner_polygon)
  samples = int(outer.length/nodeinterval+0.5)
  outer_res = np.zeros((samples,2))
  inner_res = np.zeros((samples,2))
  for i in range(samples):
    outer_res[i,:] = outer.interpolate(i/samples, normalized=True)
    inner_res[i,:] = inner.interpolate(i/samples, normalized=True)
  return outer_res, inner_res

def _repeat(
  G,
  D
):
  '''[inner-contours]
  '''
  nrows,ncols = G.shape
  Gx = np.zeros((nrows+2*D, ncols+2*D))
  Gx[D:(-D),D:(-D)]=G # core
  Gx[D:(-D),(-D):]=G[:,:D] # right
  Gx[D:(-D),:D]=G[:,(-D):] # left
  Gx[:D,D:(-D)]=G[(-D):,:] # top
  Gx[(-D):,D:(-D)]=G[:D,:] # bottom
  Gx[:D,:D]=G[(-D):,(-D):] # top-left
  Gx[:D,(-D):]=G[(-D):,:D] # top-right
  Gx[(-D):,:D]=G[:D,(-D):] # bottom-left
  Gx[(-D):,(-D):]=G[:D,:D] # bottom-right
  return Gx

def filter_offdiagonal(
  P,
  D=10,
  freq=0.4
):
  '''Filter off-diagonal weights from projection matrix
  [inner-contours]
  '''
  Px = _repeat(np.array(P), D)
  re, im = gabor(Px, theta=-np.pi/4.0, frequency=freq)
  F = np.sqrt(re**2+im**2)[D:(-D),D:(-D)]
  re, im = gabor(Px, theta=np.pi/4.0, frequency=freq)
  G = np.sqrt(re**2+im**2)[D:(-D),D:(-D)]
  P = F if np.sum(F)>np.sum(G) else G
  P = P**2 # aleviate filter low-passing
  P = np.diag(1/np.sum(P, axis=1)).dot(P)
  return P

def b_to_a_projection(
  a,
  b,
  scale
):
  '''Align polygons using OT. Both polygons have the same number of points
  [inner-contours]
  '''
  x = torch.tensor(a/scale, dtype=torch.float)
  y = torch.tensor(b/scale, dtype=torch.float)
  sinkhorn = SinkhornDistance(eps=1e-6, max_iter=200, reduction=None)
  _, P, _ = sinkhorn(x, y)
  P = filter_offdiagonal(P)
  return P

def assignments_indices(
  a, # array shape (na, 2)
  b, # array shape (nb, 2)
  P, # array shape (na, nb)
):
  '''Assignment indices
  NOTE: This function is wrong. There's a better implementation using
  mapping to angles.
  [inner-contours]
  '''
  p=np.array(P)
  x=np.array(a)
  y=np.array(b)
  y2 = np.array(list(range(len(y)))) # array shape (nb, 1)
  for i in range(len(x)):
    if len(np.unique(p[i,:]))>2:
      iy = np.argmax(p[i,:])
      y2[i] = iy
  return y2

def _assignments_proj_weighted(
  b,
  P
):
  '''Assignment by weighted average position, 1st approach
  [inner-contours]'''
  return np.diag(1/np.sum(P,axis=1)).dot(P.dot(b))

def _assignments_proj_maxcoupling(
  b,
  P
):
  '''[inner-contours]
  '''
  iy = np.argmax(P, axis=1)
  y2 = b[iy]
  return y2

def _assignments_proj_weighted2(
  a,
  b,
  P
):
  '''Assignment by weighted average projection, 2nd approach
  [inner-contours]'''
  y2 = np.diag(1/np.sum(P,axis=1)).dot(P.dot(b))
  r = y2-a # projected delta vectors
  rn = np.sqrt(np.sum(r*r,axis=1)) # norm of the deltas
  for i,_ in enumerate(a):
    ni = 0
    s = 0
    for j,_ in enumerate(b):
      if P[i, j] != 0:
        nij = np.sqrt((b[j]-a[i])**2)
        ni += P[i,j]*nij
        s += P[i,j]
    ni /= s
    y2[i] = a[i] + r[i]*ni/rn[i]
  return y2

def _lengthparam(
  line
):
  '''Parametrise line by length
  [inner-contours]
  '''
  length = []
  prevp = line[0]
  d = 0
  for _, p in enumerate(line):
    d += np.sqrt(np.sum((p-prevp)**2))
    length.append(d)
    prevp = p
  return np.array(length)

def _reparam(
  y,
  val
):
  '''Reparametrise y
  [inner-contours]
  '''
  maxval = np.max(y)
  y = y - val
  y[y>maxval/2] -= maxval
  y[y<-maxval/2] += maxval
  return y

def _assignments_proj_line(
  b,
  P
):
  '''Weighted average position over the linear ring
  [inner-contours]
  '''
  lb = LinearRing(b)
  lpb = _lengthparam(b)
  res = []
  maxc = []
  for i,_ in enumerate(b):
    Q = P[i]
    j = np.argmax(Q)
    sump = np.sum(Q)
    replb = _reparam(lpb, lpb[j])
    replb2 = Q.dot(replb)/sump + lpb[j]
    res.append(replb2)
    maxc.append(lpb[j])
  y2 = [lb.interpolate(d, normalized=False).coords[0] for d in res]
  return np.array(y2)

def assignments(
  a,
  b,
  P,
  projection_type="line"
):
  '''Assign vertices using different projection methods
  [inner-contours]
  '''
  if projection_type == "line":
    return _assignments_proj_line(b, P)
  elif projection_type == "weighted":
    return _assignments_proj_weighted(b, P)
  elif projection_type == "weighted2":
    return _assignments_proj_weighted2(a, b, P)
  return _assignments_proj_maxcoupling(b, P)

def compute_inner_contours_for_polygons_outwards(
  polygons,
  iterations=3,
  scale=5000,
  dist_cont=-5,
  node_interval=10,
):
  '''Compute inner contours for polygons starting from the
  inside and going outwards
  [inner-contours][unreferenced]
  '''
  contours_in_slice = []
  for poly in polygons:
    inner_polygon = _compute_inner_polygon_iterative(poly, iterations, dist_cont)
    if inner_polygon is None:
      continue

    outer_res, inner_res = _resample_outer_and_inner_polygons(poly, inner_polygon, node_interval)
    P = b_to_a_projection(inner_res, outer_res, scale)
    outer_final = assignments(inner_res, outer_res, P)
    contours_in_slice.append((outer_final, inner_res))
  return contours_in_slice

def compute_inner_contours_for_polygons_inwards(
  polygons,
  iterations=3,
  scale=5000,
  dist_cont=-5,
  node_interval=10
):
  '''Compute inner contours for polygons starting from the inside
  and moving outwards
  [inner-contours][unreferenced]
  '''
  contours_in_slice = []
  for poly in polygons:
    inner_polygon = _compute_inner_polygon_iterative(poly, iterations, dist_cont)
    outer_res, inner_res = _resample_outer_and_inner_polygons(poly, inner_polygon, node_interval)
    P = b_to_a_projection(outer_res, inner_res, scale)
    inner_final = assignments(outer_res, inner_res, P)
    contours_in_slice.append((outer_res, inner_final))
  return contours_in_slice
