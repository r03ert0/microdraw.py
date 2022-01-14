# microdraw.py

Functions for working with MicroDraw

Import using

```python
from microdraw import microdraw as mic
```

## FUNCTIONS

`assignments(a, b, P, projection_type='line')`

`assignments_indices(a, b, P)`

`b_to_a_projection(a, b, scale)`

`color_from_string(my_string)`
    Create a random color based on a hash of the input string

`compute_inner_contour(coords_arr, delta)`

`compute_inner_contours_for_polygons(polygons, iterations=10, projection='maxcoupling', scale=1000, nodeinterval=10, distCont=-3)`

`compute_inner_contours_for_polygons_inwards(polygons, iterations=3, scale=5000, distCont=-5, nodeinterval=10)`

`compute_inner_contours_for_polygons_outwards(polygons, iterations=3, scale=5000, distCont=-5, nodeinterval=10)`

`continuous_contours(edge_soup)`
    Obtain contiuous lines from the unordered list of edges
    in edge_soup. Returns an array of lines where each element is a
    continuous line composed of string of neighbouring vertices

`convert_polygons_to_microdraw_json(lines, name, rgb)`
    Convert an array of lines (a series of vertices) into Microdraw JSON
    format (which is the default Paper.js format)

`dataset_as_volume(dataset, filter=None)`
    Combine all regions into a single mesh. This mesh does not have triangles, only region contours.
    dataset: the dataset from which to obtain the region contours
    filter: an optional array of strings with the names of the regions
            to include in the mesh

`dataset_to_nifti(dataset, voxdim=[0.1, 0.1, 1.25], region_name=None)`
    convert dataset to nifti volume. Returns a nifti object

`download_all_regions_from_dataset(source, project, token, microdraw_url='https://microdraw.pasteur.fr')`
    Download all regions in all slices in a dataset

`download_all_regions_from_dataset_slice(source, project, slce, token, backups=False, microdraw_url='https://microdraw.pasteur.fr')`
    Download all regions in a dataset slice

`download_dataset_definition(source)`
    Download a dataset. A dataset can contain several slices, each with several regions

`download_microdraw_contours_as_polygons(source, project, sliceIndex, token, width)`
    Download microdraw contours as polygons, taking care of childrens, lines and bezier curves.

`download_project_definition(project, token)`
    Download a project. A project can contain several datasets

`draw_all_dataset(dataset, ncol=13, width=800, alpha=0.5, path=None)`
    draw all dataset

`draw_slice(v, f, slice_index, path, scale_x=3, scale_yz=0.25, slice_offset=0)`

`filter_offdiagonal(P, D=10, freq=0.4)`

`find_compound_regions(regions)`
    combine regions into compound regions

`find_contour_correspondences(manual, auto, min_pct=0.7, max_pct=1.4)`

`find_mesh_contour_to_profile_contour_correspondence(mesh_contour, registered_manual_contour, profile_contour)`

`get_points_from_segment(seg)`
    get points from segment

`get_regions_from_dataset_slice(dataset)`
    get regions from dataset slice

`icp(ref, mov)`

`icp_step(mov, ref)`

`interp(a, b, x)`
    Obtain a vector between a and b at the position a[2]<=x<=b[2]

`is_reference_orientation_a_hole(regions, sub_regions)`
    determine the orientation of sub regions in a compound region

`load_dataset(path)`
    load dataset stored in json format

`no_duplicates_contour(vert_point_and_coord, contour)`
    Remove duplicate vertices from the contour given by vertices
    `vert_point_and_coord` and edges in `contour`. The indices in
    `contour` are re-indexed accordingly.

`paperjs_path_to_polygon(co)`
    Convert Paper.js paths to a polygon (a line composed of consecutive vertices).
    
    Note: Paper.js Bézier curves are encoded as px, py, ax, ay, bx, by,
    where px, py is an anchor point, ax, ay is the previous handle and
    bx, by the next handle. SVG path tools use a more standard encoding:
    p1x,p1y, b1x, b1y, a2x, a2y, p2x, p2y, where p1x, p1y is the start
    anchor point, p2x, p2y the end anchor point, b1x, b1y is the
    handle coming out from p1x, p1y, and a2x, a2y is the handle entering
    into the end anchor point.

`raw_contour(v, f, x)`
    Obtain the contour produced by slicing the mesh with vertices v
    and faces f at coordinate x in the last dimension. Produces a
    list of vertices and a list of edges. The vertices for each
    edge are unique which means that vertices at connecting edges
    will be duplicated. Each vertex includes a reference to the
    edge where it comes from. This reference contains the index
    of the triangle, the number of the edge withing the triangle
    (0, 1 or 2), and the distance from the beginning of the edge.

`register_contours(manual, auto, corresp)`

`register_microdraw_contours_to_mesh_contours_for_slice(source, project, sliceIndex, token, v, f, width, height, scale_x, scale_yz, slice_offset)`

`save_contour(path, ver, con)`
    A contour is a list of vertices and a list of edges. A contour can
    contain multiple independent closed contours, but the list of vertices
    and edges is always unique. This function saves the contour in a
    format readable by MeshSurgery.

`save_dataset(data, path)`
    save dataset in json format

`save_dataset_as_nifti(dataset, path, voxdim=[0.1, 0.1, 1.25], region_name=None)`
    save dataset as a nifti volume

`save_dataset_as_text_mesh(dataset, path, voxdim=[0.1, 0.1, 1.25])`
    save dataset as a text mesh

`save_slice_contours_in_svg(path, contours, width, height)`

`scale_contours_to_image(v, width, height, scale_yz)`

`slice_mesh(v, f, z, min_contour_length=10)`
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

## Data types

* <b>Annotation:</b> Any type of data added by the user, in particular, regions
* <b>Bézier curve</b>: A curve composed of a series of polynomial curves, each determined by a start anchor, an end anchor, and two control points.
* <b>Color:</b> A combination of red, green and blue components, plus an alpha value, encoded in hexadecimal.
* <b>Contour, continuous:</b>
* <b>Contour, raw:</b>
* <b>Contour:</b> Same as a region?
* <b>Dataset:</b> Annotations for a series of slices
* <b>Image:</b> The image being annotated, encoded in .dzi format.
* <b>Inner contour:</b>
* <b>Inner polygon:</b>
* <b>Label:</b> A name and a color.
* <b>Mesh contour:</b> Contour obtained by slicing a mesh.
* <b>Mesh:</b> Collection of vertices and triangles.
* <b>Microdraw contour:</b>
* <b>Microdraw JSON:</b>
* <b>Nifti:</b> Volume encoded in Nifti 1 format, contains the volume data and metadata.
* <b>Outer contour:</b>
* <b>Outer polygon:</b>
* <b>Paper.js path:</b> A simple or compound path encoded in Paper.js format.
* <b>Point:</b> A position in space described by its coordinates.
* <b>Polygon</b>: A curve composed of straight lines.
* <b>Profile contour:</b> A contour used for computing profiles.
* <b>Profile:</b> Grey level profile along a line going from the outer contour to the inner contour.
* <b>Project:</b> A Microdraw project, containing a dataset plus information about collaborators, permissions, list of features to annotate, project description, project URL.
* <b>Region:</b> a path enclosing a part of the image. Can be a simple path or a compound path. Can be a Bézier curve or a polygon. It is associated to a label.
* <b>Segment:</b>
* <b>Slice (noun):</b> An image and its annotations, identified by the index in the dataset it belongs.
* <b>Slice (verb):</b> To intersecting a volumetric object with a plane.
* <b>SVG:</b> A standard format for encoding 2D vectorial graphics.
* <b>Text mesh:</b> A mesh encoded in plain text. The first row contains number of vertices followed by number of triangles. The following rows contain the x, y and z coordinates for each of the vertices. The next rows contain each of the mesh triangles, where each triangle contains the indices of its 3 vertices. The 1st vertex has index 0.
* <b>Volume:</b> Same as Nifti?
