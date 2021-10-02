import sys
import numpy as np
from scipy.spatial import distance_matrix

def _interp(a, b, x):
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
  contour = []
  vert_point_and_coord = []
  for i in range(len(f)):
    a,b,c = f[i]
    ed = []
    tri_based_coord = []

    if np.abs(v[a,2]-x)<EPS:
      ed.append(v[a])
      tri_based_coord.append((i,0,0)) # triangle i, edge #0, at the beginning
    if np.abs(v[b,2]-x)<EPS:
      ed.append(v[b])
      tri_based_coord.append((i,1,0)) # triangle i, edge #1, at the beginning
    if np.abs(v[c,2]-x)<EPS:
      ed.append(v[c])
      tri_based_coord.append((i,2,0)) # triangle i, edge #2, at the beginning
    if (v[a,2]-x)*(v[b,2]-x) < 0:
      p,t = _interp(v[a,:],v[b,:],x)
      ed.append(p)
      tri_based_coord.append((i,0,t)) # triangle i, edge #0, t% of the length
    if (v[b,2]-x)*(v[c,2]-x) < 0:
      p,t = _interp(v[b,:],v[c,:],x)
      ed.append(p)
      tri_based_coord.append((i,1,t)) # triangle i, edge #1, t% of the length
    if (v[c,2]-x)*(v[a,2]-x) < 0:
      p,t=_interp(v[c,:],v[a,:],x)
      ed.append(p)
      tri_based_coord.append((i,2,t)) # triangle i, edge #2, t% of the length

    if len(ed) == 2:
      n = len(vert_point_and_coord)
      contour.append((n, n+1))
      vert_point_and_coord.append((ed[0],tri_based_coord[0]))
      vert_point_and_coord.append((ed[1],tri_based_coord[1]))
    elif len(ed)>0:
      print("WEIRD EDGE", ed, tri_based_coord)

  contour=np.array(contour)
  return (vert_point_and_coord, contour)

def no_duplicates_contour(vert_point_and_coord, contour):
  '''
  Remove duplicate vertices from the contour given by vertices
  `vert_point_and_coord` and edges in `contour`. The indices in
  `contour` are re-indexed accordingly.
  '''
  # distance from each point to the others
  ver = np.zeros((len(vert_point_and_coord),3))
  for i in range(len(vert_point_and_coord)):
    ver[i,:] = vert_point_and_coord[i][0]
  m = distance_matrix(ver, ver)

  # set the diagonal to a large number
  m2 = m + np.eye(m.shape[0])*(np.max(m)+1)

  # for each point, find the closest among the others
  closest = np.argmin(m2, axis=0)
  
  lut = [i for i in range(len(ver))]

  # make a list of unique vertices and a look-up table
  n=0
  unique_vert_point_and_coord = []
  for i in range(len(ver)):
    if i<closest[i]:
      unique_vert_point_and_coord.append((ver[i],vert_point_and_coord[i][1]))
      lut[i] = n
      lut[closest[i]] = n
      n+=1

  # re-index the edges to refer to the new list of unique vertices
  for i in range(len(contour)):
    contour[i] = (lut[contour[i,0]], lut[contour[i,1]])
  
  return (unique_vert_point_and_coord, contour)

def continuous_contours(edge_soup):
  '''
  Obtain contiuous lines from the unordered list of edges
  in edge_soup. Returns an array of lines where each element is a
  continuous line composed of string of neighbouring vertices
  '''
  co1=edge_soup.copy()
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

def slice_mesh(v, f, z, min_contour_length=10):
  '''
  Slices the mesh of vertices v and faces f with the plane
  of given z coordinate. Returns:
  * unique_verts_point_and_coord: a list of unique vertices,
  * mesh_relative_vertex_coords: their coordinates relative to the
    mesh. Each row has 3 values: index of the mesh triangle that
    was sliced, index of the edge within that triangle, position of
    the vertex within that edge. The value of the position of the
    vertex within the edge is 0 if the vertex is at the beginning of
    the edge, and 1 if it is at the end.
  * edges: a list of edges,
  * lines: and a list of continuous lines.
  '''
  raw_verts, raw_cont = raw_contour(v, f, z)
  if len(raw_cont)<min_contour_length:
      return None,None,None,None

  unique_verts_point_and_coord, edges = no_duplicates_contour(raw_verts, raw_cont)

  lines = [line for line in continuous_contours(edges) if len(line)>=min_contour_length]

  unique_verts = np.zeros((len(unique_verts_point_and_coord),3))
  mesh_relative_vertex_coords = []
  for i in range(len(unique_verts_point_and_coord)):
      unique_verts[i,:] = unique_verts_point_and_coord[i][0]
      mesh_relative_vertex_coords.append(unique_verts_point_and_coord[i][1])
  return unique_verts, mesh_relative_vertex_coords, edges, lines

def scale_contours_to_image(v, width, height, scale_yz):
  s = [[ve[0]/scale_yz,height-ve[1]/scale_yz] for ve in v]
  return np.array(s)

