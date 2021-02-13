# microdraw.py
Python package for working with microdraw

Import using

```python
from microdraw import microdraw as mic
```

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
