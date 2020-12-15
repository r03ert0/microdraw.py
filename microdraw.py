"""microdraw.py: functions for working with MicroDraw"""

from layers import SinkhornDistance
from scipy.spatial import distance_matrix
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon, MultiPolygon, LinearRing
from shapely.ops import nearest_points
from skimage import io
from skimage.draw import polygon, polygon_perimeter
from svgpathtools import Path, CubicBezier, Line
from tqdm import tqdm
import compute_inner_contours as cic
import contourOptimalTransportToolbox as cOTT
import contourToolbox as cT
import generateProfilesToolbox as gpT
import hashlib
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import sys
import torch
import urllib.request as urlreq
import pyclipper
from skimage.filters import gabor


def download_project_definition(project, token):
  '''Download a project. A project can contain several datasets'''

  url = "https://microdraw.pasteur.fr/project/json/%s?token=%s"%(project, token)
  res = urlreq.urlopen(url)
  txt = res.read()
  prj = json.loads(txt)

  return prj

def download_dataset_definition(source):
  '''Download a dataset. A dataset can contain several slices, each with
     several regions'''

  res = urlreq.urlopen(source)
  txt = res.read()
  prj = json.loads(txt)

  return {
      "pixelsPerMeter": prj["pixelsPerMeter"],
      "numSlices": len(prj["tileSources"]),
      "project": prj
      }

def download_all_regions_from_dataset_slice(source, project, slce, token, backups=False):
  '''Download all regions in a dataset slice'''

  url = "https://microdraw.pasteur.fr/api?source=%s&project=%s&slice=%s&token=%s"%(
      source,
      project,
      slce,
      token)

  if backups is True:
    url = url + "&backup"

  res = urlreq.urlopen(url)
  txt = res.read()
  conts = json.loads(txt)

  return conts

def download_all_regions_from_dataset(source, project, token):
  '''Download all regions in all slices in a dataset'''

  dataset = download_dataset_definition(source)
  dataset["slices"] = []
  for i in tqdm(range(dataset["numSlices"])):
    dataset["slices"].append(download_all_regions_from_dataset_slice(source, project, i, token))
  return dataset

def get_points_from_segment(seg):
  '''get points from segment'''

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

def get_regions_from_dataset_slice(dataset):
  '''get regions from dataset slice'''

  regions = []
  for region in dataset:
    name = region['annotation']['name']
    path_type = region['annotation']['path'][0]
    if path_type == "Path" and 'segments' in region['annotation']['path'][1]:
      seg = np.asfortranarray(region['annotation']['path'][1]['segments'])
      points = get_points_from_segment(seg)
      regions.append((name, np.array(points)))
    elif path_type == "CompoundPath":
      children = region['annotation']['path'][1]['children']
      for child in children:
        if 'segments' in child[1]:
          segments = [np.asfortranarray(child[1]['segments'])]
          for seg in segments:
            points = get_points_from_segment(seg)
            regions.append((name, np.array(points)))
  return regions

def draw_all_dataset(dataset, ncol=13, width=800, alpha=0.5, path=None):
  '''draw all dataset'''

  plt.figure(figsize=(25, 10))
  i = 0
  j = 0
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region in regions:
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

def dataset_as_volume(dataset):
  '''get dataset as a volume'''

  verts = []
  eds = []
  neds = 0
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for _, region in regions:
      verts.extend([(x, y, slce) for x, y in region])
      eds.extend([(neds+i, neds+(i+1)%len(region)) for i in range(len(region))])
      neds = len(eds)
  return verts, eds

def save_dataset_as_text_mesh(dataset, path):
  '''save dataset as a text mesh'''

  verts, eds = dataset_as_volume(dataset)
  mesh = "%i 0 %i"%(len(verts), len(eds))
  verts_str = "\n".join(["%f %f %f"%(x*0.1, y*0.1, z*1.25) for x, y, z in verts])
  eds_str = "\n".join(["%i %i"%(i, j) for i, j in eds])
  mesh = "\n".join((mesh, verts_str, eds_str))

  file = open(path, 'w')
  file.write(mesh)

def dataset_to_nifti(dataset, voxdim=[0.1, 0.1, 1.25], region_name=None):
  '''convert dataset to nifti volume'''

  verts, _ = dataset_as_volume(dataset)
  vmin, vmax = np.min(verts, axis=0), np.max(verts, axis=0)
  vmin = np.floor(vmin)
  vmax = np.ceil(vmax)
  size = vmax-vmin
  size[2] = dataset["numSlices"]
  img = np.zeros([int(x) for x in size], 'uint8')
  for slce in range(dataset["numSlices"]):
    regions = get_regions_from_dataset_slice(dataset["slices"][slce])
    for name, region in regions:
      if region_name is None or (region_name is not None and name == region_name):
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
  nii = nib.Nifti1Image(img, affine=affine)

  return nii

def save_dataset_as_nifti(dataset, path, voxdim=[0.1, 0.1, 1.25], region_name=None):
  '''save dataset as a nifti volume'''

  nii = dataset_to_nifti(dataset, voxdim, region_name)
  nib.save(nii, path)

def save_dataset(data, path):
  '''save dataset in json format'''

  with open(path, 'w') as outfile:
    json.dump(data, outfile)

def color_from_string(my_string):
  '''create a random color based on a hash of the input string'''

  sha = hashlib.sha256()
  sha.update(my_string.encode())
  return "#" + sha.hexdigest()[:6]

def save_contour(path, ver, con):
    '''
    A contour is a list of vertices and a list of edges. A contour can
    contain multiple independent closed contours, but the list of vertices
    and edges is always unique. This function saves the contour in a
    format readable by MeshSurgery.
    '''
    fi=open(path, "w")
    fi.write("%i 0 %i\n"%(len(ver),len(con)))
    fi.write("\n".join(["%f %f 0"%(cx,cy) for (cx,cy,_) in ver]) + "\n")
    fi.write("\n".join([str(a)+" "+str(b) for (a,b) in con]))
    fi.close()

def interp(a,b,x):
    '''
    Obtain a vector between a and b at the position a[2]<=x<=b[2]
    '''
    l = b[2]-a[2]
    t = (x-a[2])/l
    p = a + t*(b-a)
    return p,t

def raw_contour(v, f, x):
    '''
    Obtain the contour produced by slicing the mesh with vertices v
    and faces f at coordinate x in the last dimension. Produces a
    list of vertices and a list of edges. The vertices for each
    edge are unique which means that vertices at connecting edges
    will be duplicated. Each vertex includes a reference to the
    edge where it comes from. This reference contains the index
    of the triangle, the number of the edge withing the triangle
    (0, 1 or 2), and the distance from the beginning of the edge.
    '''
    EPS = sys.float_info.epsilon
    con = []
    verpc = []
    for i in range(len(f)):
        a,b,c = f[i]
        ed = []
        co = []

        if np.abs(v[a,2]-x)<EPS:
            ed.append(v[a])
            co.append((i,0,0)) # triangle i, edge #0, at the beginning
        if np.abs(v[b,2]-x)<EPS:
            ed.append(v[b])
            co.append((i,1,0)) # triangle i, edge #1, at the beginning
        if np.abs(v[c,2]-x)<EPS:
            ed.append(v[c])
            co.append((i,2,0)) # triangle i, edge #2, at the beginning
        if (v[a,2]-x)*(v[b,2]-x) < 0:
            p,t = interp(v[a,:],v[b,:],x)
            ed.append(p)
            co.append((i,0,t)) # triangle i, edge #0, t% of the length
        if (v[b,2]-x)*(v[c,2]-x) < 0:
            p,t = interp(v[b,:],v[c,:],x)
            ed.append(p)
            co.append((i,1,t)) # triangle i, edge #1, t% of the length
        if (v[c,2]-x)*(v[a,2]-x) < 0:
            p,t=interp(v[c,:],v[a,:],x)
            ed.append(p)
            co.append((i,2,t)) # triangle i, edge #2, t% of the length

        if len(ed) == 2:
            n = len(verpc)
            con.append((n, n+1))
            verpc.append((ed[0],co[0]))
            verpc.append((ed[1],co[1]))
        elif len(ed)>0:
            print("WEIRD ED", ed,co)

    con=np.array(con)
    return (verpc, con)

def continuous_contour(verpc, con):
    '''
    Remove duplicate vertices from the contour given by vertices ver
    and edges con. The indices in con are re-indexed accordingly.
    '''
    # distance from each point to the others
    ver = np.zeros((len(verpc),3))
    for i in range(len(verpc)):
        ver[i,:] = verpc[i][0]
    m = distance_matrix(ver, ver)

    # set the diagonal to a large number
    m2 = m + np.eye(m.shape[0])*(np.max(m)+1)

    # for each point, find the closest among the others
    closest = np.argmin(m2, axis=0)
    
    lut = [i for i in range(len(ver))]

    # make a list of unique vertices and a look-up table
    n=0
    uverpc = []
    for i in range(len(ver)):
        if i<closest[i]:
            uverpc.append((ver[i],verpc[i][1]))
            lut[i] = n
            lut[closest[i]] = n
            n+=1

    # re-index the edges to refer to the new list of unique vertices
    for i in range(len(con)):
        con[i] = (lut[con[i,0]], lut[con[i,1]])
    
    return (uverpc, con)

def sort_contours(co):
    '''
    Obtain contiuous lines from the unordered list of edges
    in co. Returns an array of lines where each element is a
    continuous line composed of string of neighbouring vertices
    '''
    co1=co.copy()
    lines = []
    while True:
        line = []
        start = co1[0,0]
        line.append(start)
        while True:
            found = 0
            for i,(a,b) in enumerate(co1):
                if a == line[-1]:
                    line.append(b)
                    found = 1
                elif b == line[-1]:
                    line.append(a)
                    found = 1
                if found:
                    co1 = np.delete(co1,i,0)
                    break
            if found == 0:
                break
        if len(line):
            lines.append(line)
        else:
            break
        if len(co1) == 0:
            break
    return lines

def slice_mesh(v, f, x, min_contour_length=10):
    '''
    Slices the mesh of vertices v and faces f with the plane
    of last coordinate x. Returns a list of unique vertices,
    their coordinates relative to the mesh,
    a list of edges, and a list of continuous lines.
    '''
    ver, con = raw_contour(v, f, x)
    if len(con)<min_contour_length:
        return None,None,None,None
    uverpc, ccon = continuous_contour(ver, con)

    cons = sort_contours(ccon)
    uver = np.zeros((len(uverpc),3))
    vercoords = []
    for i in range(len(uverpc)):
        uver[i,:] = uverpc[i][0]
        vercoords.append(uverpc[i][1])
    return uver, vercoords, ccon, cons

def scale_contours_to_image(v, width, height, scale_yz):
    s = [[ve[0]/scale_yz,height-ve[1]/scale_yz] for ve in v]
    return np.array(s)

def draw_slice(v, f, slice_index, path, scale_x=3, scale_yz=0.25, slice_offset=0):
    his = io.imread(path)
    iH, iW, _ = his.shape
    ve, _, _, _ = slice_mesh(v, f, (slice_index-slice_offset)*scale_x)
    ve = scale_contours_to_image(ve, iW, iH, scale_yz)
    plt.imshow(his)
    plt.scatter(ve[:,0],ve[:,1])

def paperjs_path_to_polygon(co):
    '''
    Convert Paper.js paths to a polygon (a line composed of
    consecutive vertices).
    
    Note: Paper.js Bézier curves are encoded as px, py, ax, ay, bx, by,
    where px, py is an anchor point, ax, ay is the previous handle and
    bx, by the next handle. SVG path tools use a more standard encoding:
    p1x,p1y, b1x, b1y, a2x, a2y, p2x, p2y, where p1x, p1y is the start
    anchor point, p2x, p2y the end anchor point, b1x, b1y is the
    handle coming out from p1x, p1y, and a2x, a2y is the handle entering
    into the end anchor point.
    '''
    mysegs = []
    for i in range(len(co)):
        c = co[i%len(co)]
        c1 = co[(i+1)%len(co)]
        try:
            if isinstance(c[0], list) or isinstance(c[0], np.ndarray):
                # print("c[0] is list")
                [s,sj],[_,_],[b,bj] = c
            else:
                # print("c[0] is not list:", type(c[0]))
                s,sj,b,bj = c[0],c[1],0,0
            if isinstance(c1[0], list) or isinstance(c1[0], np.ndarray):
                # print("c1[0] is list")
                [s1,s1j],[a1,a1j],[_,_] = c1
            else:
                # print("c1[0] is not list:", type(c1[0]))
                s1,s1j,a1,a1j = c1[0],c1[1],0,0
            # print("c:", c)
            # print("c1:", c1)
            # print("s,sj:",s,sj,", b,bj:",b,bj, ", s1,sij:", s1,s1j, ", a1,a1j:",a1,a1j)
            seg = CubicBezier(complex(s,sj), complex(s+b,sj+bj), complex(s1+a1,s1j+a1j), complex(s1,s1j))
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

def download_microdraw_contours_as_polygons(source, project, sliceIndex, token, width):
    conts = download_all_regions_from_dataset_slice(source, project, sliceIndex, token)
    if len(conts) == 0:
        # print("len(conts) is 0")
        return
    polys = []
    for i in range(len(conts)):
        root = conts[i]["annotation"]["path"][1]

        if "segments" in root:
            poly = paperjs_path_to_polygon(np.array(root["segments"]))
            if poly is not None:
                poly = poly*width/1000 # scale to match image size
                polys.append(poly)
            # else:
            #     print("segments poly is None")
            #     print(poly)
        if "children" in root:
            for cont in root["children"]:
                poly = paperjs_path_to_polygon(np.array(cont[1]["segments"]))
                if poly is not None:
                    poly = poly*width/1000 # scale to match image size
                    polys.append(poly)
                # else:
                #    print("children poly is none")
    return polys

def save_slice_contours_in_svg(path, contours, width, height):
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
    file = open(path,"w")
    file.write(svg)
    file.close()

def convert_polygons_to_microdraw_json(lines, name, rgb):
    '''
    Convert an array of lines (a series of vertices) into Microdraw JSON
    format (which is the default Paper.js format)
    '''
    text = "[\n"
    annotations = []
    for line in lines:
        annotation = '''{
        "annotation":{"path":["Path",{"applyMatrix":true,"segments":%s,"closed":false,"fillColor":[%s],"strokeColor":[0,0,0],"strokeScaling":false}
        ],"name":"%s"}}'''%(line,rgb,name)
        annotations.append(annotation)
    text += ",\n".join(annotations)
    text += "]"
    return text

def icp_step(mov, ref):
    out = []
    for pt in mov:
        out.append(nearest_points(Polygon(ref),Point(pt))[0].coords[0])
    out = np.array(out)
    oout=Polygon(out).centroid.coords[0]
    omov=Polygon(mov).centroid.coords[0]
    C = np.zeros([2,2])
    for i in range(0,len(mov)):
        V1 = np.array([mov[i]-omov])
        V2 = np.array([out[i]-oout])
        C = C + V1.T*V2
    U,_,V=np.linalg.svd(C)
    R=U.dot(V)
    t=omov-R.dot(oout)
    mov = mov.dot(R)
    mov = mov - t
    dist = np.arccos((np.trace(R)-1)/2.0)
    dist = dist + np.linalg.norm(t)
    return mov, dist

def icp(ref, mov):
    maxiter=100
    tol=1e-6
    dist0=-1
    for i in range(maxiter):
        mov,dist=icp_step(mov, ref)
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

def find_contour_correspondences(manual, auto):
    malen = len(manual)
    aulen = len(auto)
    m = np.ones((malen,aulen))*(-1)

    for i in range(malen):
        for j in range(aulen):
            pola = Polygon(manual[i]).convex_hull
            polb = Polygon(auto[j]).convex_hull
            areaa = pola.area
            areab = polb.area
            areaab = pola.intersection(polb).area
            if areaab>0:
                if areaa/areaab<0.7 or areab/areaab<0.7 or areaa/areaab>1.4 or areab/areaab>1.4:
                    areaab = -1
            else:
                areaab = -1
            m[i,j] = areaab
    corresp = np.argmax(m.T,axis=0)
    for i,val in enumerate(corresp):
        if m[i,val] == -1:
            corresp[i] = -1
    return corresp

def register_contours(manual, auto, corresp):
    # print("corresp", corresp, corresp.shape)
    # print("malen, aulen:", len(manual), len(auto))
    registered = []
    for imanual,iauto in enumerate(corresp):
        print("Registering manual contour %i to automatic contour %i"%(imanual, iauto))
        mov = manual[imanual]
        if iauto<0:
            print("Manual contour %i does not have a corresponding automatic contour"%(imanual))
            registered.append(mov)
            continue
        ref = auto[iauto]
        mov2,_,_ = icp(ref, mov)
        registered.append(mov2)
    return registered

def register_microdraw_contours_to_mesh_contours_for_slice(source, project, sliceIndex,
        token, v, f, width, height, scale_x, scale_yz, slice_offset):
    manual = download_microdraw_contours_as_polygons(source, project, sliceIndex, token, width)
    if manual is None:
        # print("manual is None")
        return None,None,None,None
    ve, veco, _, lines = slice_mesh(v, f, (sliceIndex-slice_offset)*scale_x)
    if ve is None:
        return None,None,None,None
    ve = scale_contours_to_image(ve, width, height, scale_yz)
    auto = [ve[line] for line in lines]
    autoco = []
    for line in lines:
        one_autoco = []
        for i in line:
            one_autoco.append(veco[i])
        autoco.append(one_autoco)
    if len(manual)==0 or len(auto) ==0:
        # print("either len(manual) or len(auto) are 0:", len(manual), len(auto))
        return None,None,None,None
    corresp = find_contour_correspondences(manual, auto)
    registered = register_contours(manual, auto, corresp)
    auto2 = []
    autoco2 = []
    for i in range(len(corresp)):
        if corresp[i] >= 0:
            auto2.append(auto[int(corresp[i])])
            autoco2.append(autoco[int(corresp[i])])
    return registered, manual, auto2, autoco2

def find_mesh_contour_to_profile_contour_correspondence(
    mesh_contour, # the contour obtained by slicing the mesh: its points refer back to mesh vertices
    registered_manual_contour, # the manually drawn contour from microdraw, registered to the mesh
    profile_contour # the manually drawn contour from microdraw, non-registered, downsampled, used to extract profiles
):
    # align registered_manual_contour to mesh_contour
    # the result is ind: a list giving for each point in registered_manual_contour the index of the closest point in mesh_contour
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

    regm2reg = np.zeros(nsamples_mesh,dtype=int) # correspondence between resample registered and raw registered
    reg2prof = np.zeros(nsamples_regm,dtype=int) # correspondence between raw registered and profile_contour

    for j in range(nsamples_regm):
        le = regm_ring.project(Point(registered_manual_contour[j,:]),normalized=True)
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

def compute_inner_contours_for_polygons(polygons,
    iterations=10,
    projection="maxcoupling",
    scale=1000,
    nodeinterval=10,
    distCont=-3
    ):
    return cic.compute_inner_contours_for_polygons(
        polygons,
        iterations=iterations,
        projection=projection,
        scale=scale,
        nodeinterval=nodeinterval,
        distCont=distCont)

def compute_inner_contour(coords_arr, delta):
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

def _compute_inner_polygon_iterative(polygon, iterations, distCont):
    # compute inner polygon in steps
    inner_polygon = polygon
    for _ in range(iterations):
        inner_polygon = cOTT.computeInnerContour(inner_polygon, distCont)
    return inner_polygon

def _resample_outer_and_inner_polygons(outer_polygon, inner_polygon, nodeinterval):
    # resample inner and outer polygons
    outer = LinearRing(outer_polygon)
    inner = LinearRing(inner_polygon)
    samples = int(outer.length/nodeinterval+0.5)
    outer_res = np.zeros((samples,2))
    inner_res = np.zeros((samples,2))
    for i in range(samples):
        outer_res[i,:] = outer.interpolate(i/samples, normalized=True)
        inner_res[i,:] = inner.interpolate(i/samples, normalized=True)
    return outer_res, inner_res

def _repeat(G, D):
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

def filter_offdiagonal(P, D=10, freq=0.4):
    # filter off-diagonal weights from projection matrix
    Px = _repeat(np.array(P), D)
    re, im = gabor(Px, theta=-np.pi/4.0, frequency=freq)
    F = np.sqrt(re**2+im**2)[D:(-D),D:(-D)]
    re, im = gabor(Px, theta=np.pi/4.0, frequency=freq)
    G = np.sqrt(re**2+im**2)[D:(-D),D:(-D)]
    P = F if np.sum(F)>np.sum(G) else G
    P = P**2 # aleviate filter low-passing
    P = np.diag(1/np.sum(P, axis=1)).dot(P)
    return P

def b_to_a_projection(a, b, scale):
    # align polygons using OT. Both polygons have the same number of points
    x = torch.tensor(a/scale, dtype=torch.float)
    y = torch.tensor(b/scale, dtype=torch.float)
    sinkhorn = SinkhornDistance(eps=1e-6, max_iter=200, reduction=None)
    _, P, _ = sinkhorn(x, y)
    P = filter_offdiagonal(P)
    return P

def assignments_indices(a, b, P):
    p=np.array(P)
    x=np.array(a)
    y=np.array(b)
    y2 = np.array([i for i in range(len(y))])
    for i in range(len(x)):
        if len(np.unique(p[i,:]))>2:
            iy = np.argmax(p[i,:])
            y2[i] = iy
    return y2

def _assignments_proj_weighted(b, P):
    '''weighted average position'''
    return np.diag(1/np.sum(P,axis=1)).dot(P.dot(b))

def _assignments_proj_maxcoupling(b, P):
    iy = np.argmax(P, axis=1)
    y2 = b[iy]
    return y2

def _assignments_proj_weighted2(a, b, P):
    y2 = np.diag(1/np.sum(P,axis=1)).dot(P.dot(b))
    r = y2-a # projected delta vectors
    rn = np.sqrt(np.sum(r*r,axis=1)) # norm of the deltas
    for i,_ in enumerate(a):
        ni = 0
        s = 0
        for j,_ in enumerate(b):
            if P[i,j] != 0:
                nij = np.sqrt((b[j]-a[i])**2)
                ni += P[i,j]*nij
                s += P[i,j]
        ni /= s
        y2[i] = a[i] + r[i]*ni/rn[i]
    return y2

def _lengthparam(line):
    length = []
    prevp = line[0]
    d = 0
    for _,p in enumerate(line):
        d += np.sqrt(np.sum((p-prevp)**2))
        length.append(d)
        prevp = p
    return np.array(length)

def _reparam(y,val):
    maxval = np.max(y)
    y = y-val
    y[y>maxval/2]-=maxval
    y[y<-maxval/2]+=maxval
    return y

def _assignments_proj_line(b, P):
    '''weighted average position over the linear ring'''
    lb = LinearRing(b)
    lpb = _lengthparam(b)
    res = []
    maxc = []
    for i,_ in enumerate(b):
        Q=P[i]
        j = np.argmax(Q)
        sump = np.sum(Q)
        replb = _reparam(lpb, lpb[j])
        replb2 = Q.dot(replb)/sump + lpb[j]
        res.append(replb2)
        maxc.append(lpb[j])
    y2 = []
    for d in res:
        y2.append(lb.interpolate(d, normalized=False).coords[0])
    return np.array(y2)

def assignments(a, b, P, projection_type="line"):
    b_final = None
    if projection_type is "line":
        b_final = _assignments_proj_line(b, P)
    elif projection_type is "weighted":
        b_final = _assignments_proj_weighted(b, P)
    elif projection_type is "weighted2":
        b_final = _assignments_proj_weighted2(a, b, P)
    else:
        b_final = _assignments_proj_maxcoupling(b, P)
    return b_final

def compute_inner_contours_for_polygons_outwards(polygons,
    iterations=3,
    scale=5000,
    distCont=-5,
    nodeinterval=10,
    ):
    contours_in_slice = []
    for polygon in polygons:
        
        inner_polygon = _compute_inner_polygon_iterative(polygon, iterations, distCont)
        outer_res, inner_res = _resample_outer_and_inner_polygons(polygon, inner_polygon, nodeinterval)
        P = b_to_a_projection(inner_res, outer_res, scale)
        outer_final = assignments(inner_res, outer_res, P)
        contours_in_slice.append((outer_final, inner_res))
    return contours_in_slice

def compute_inner_contours_for_polygons_inwards(polygons,
    iterations=3,
    scale=5000,
    distCont=-5,
    nodeinterval=10
    ):
    contours_in_slice = []
    for polygon in polygons:
        inner_polygon = _compute_inner_polygon_iterative(polygon, iterations, distCont)
        outer_res, inner_res = _resample_outer_and_inner_polygons(polygon, inner_polygon, nodeinterval)
        P = b_to_a_projection(outer_res, inner_res, scale)
        inner_final = assignments(outer_res, inner_res, P)
        contours_in_slice.append((outer_res, inner_final))
    return contours_in_slice