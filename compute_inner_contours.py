# Script to generate inner contours.
# Outer contours from MicroDraw have been saved as json files, will be loaded here, and matched to generated inner contours.
# Note: This script requires polygon contours (not Bezier curves)


# Imports
import numpy as np
import torch
from layers import SinkhornDistance
import contourOptimalTransportToolbox as cOTT
import contourToolbox as cT

def compute_inner_contours_for_polygons(polygons,
    iterations=10,
    projection="maxcoupling",
    scale=1000,
    nodeinterval=10,
    distCont=-3
    ):
    contours_in_slice = []
    for polygon in polygons:
        # Generate inner contours
        inner_polygon = polygon
        for i in range(iterations):
            inner_polygon = cOTT.computeInnerContour(inner_polygon, distCont)

        # Resample contours
        x_ref,y_ref,_,_ = cOTT.resampleContour(polygon, nodeinterval)
        x_mov,y_mov,_,_ = cOTT.resampleContour(inner_polygon, nodeinterval)

        # Resize contours, so that sinkhorn OT does not saturate
        N = np.maximum(len(x_ref), len(x_mov))

        ref = np.zeros((N,2))
        ref[:len(x_ref),0] = x_ref/scale
        ref[:len(x_ref),1] = y_ref/scale

        mov = np.zeros((N,2))
        mov[:len(x_mov),0] = x_mov/scale
        mov[:len(x_mov),1] = y_mov/scale

        x = torch.tensor(ref, dtype=torch.float)
        y = torch.tensor(mov, dtype=torch.float)

        sinkhorn = SinkhornDistance(eps=1e-5, max_iter=100, reduction=None)
        dist, P, _ = sinkhorn(x, y)
        print("Sinkhorn distance: {:.3f}".format(dist.item()))

        # Projection : for each point on the outer contour there is only one corresponding point on the inner contour (picked as the optimal point with the highest coupling)
        # c_o,c_p = cOTT.show_assignments_proj_maxcoupling(x, y, P, False)
        # c_o,c_p = cOTT.show_assignments_proj(x, y, P, False)
        c_o = None
        c_p = None
        if projection == "weighted":
            c_o,c_p = cOTT.show_assignments_proj(x, y, P, False)
        elif projection == "weighted2":
            c_o,c_p = cOTT.show_assignments_proj2(x, y, P, False)
        else:
            c_o,c_p = cOTT.show_assignments_proj_maxcoupling(x, y, P, False)

        # Resize them back
        c_orig = c_o*scale
        c_proj = c_p*scale

        contours_in_slice.append((c_orig, c_proj,len(x_ref),len(x_mov)))
    return contours_in_slice
