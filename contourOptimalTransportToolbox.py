from email.policy import default
from scipy.interpolate import splev, splrep
import bezier
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
from shapely.geometry import LinearRing

def extractNodesFromJsonBezier(jsonPath,roi): ### for bezier curves
    ##read json file
    with open(jsonPath) as f:
        data = json.load(f)

    for i in range(len(data['Regions'])):
        if data['Regions'][i]['name'] == roi:
            print(data['Regions'][i]['name'],'OK')
            ## extract node information for one ROI
            nodes = np.asfortranarray(data['Regions'][i]['path'][1]['segments'])

            ## compute and organize the control points and anchor points
            coord = [[],[]]
            for i in range(nodes.shape[0]):
                coord[0].append(nodes[i,0,0]+nodes[i,1,0])
                coord[0].append(nodes[i,0,0])
                coord[0].append(nodes[i,0,0]+nodes[i,2,0])
                coord[1].append(nodes[i,0,1]+nodes[i,1,1])
                coord[1].append(nodes[i,0,1])
                coord[1].append(nodes[i,0,1]+nodes[i,2,1])
            points = np.asarray(coord)
            return points
    print('ROI not found')

def extractNodesFromJsonPolygon(jsonPath):
    ##read json file
    with open(jsonPath) as f:
        data = json.load(f)

    #### read polygon
    contours_pts = []
    for i in range(len(data)):
        ## extract node information for one ROI
        nodes = np.asfortranarray(data[i]['annotation']['path'][1]['segments'])
        coord = [[],[]]
        for i in range(nodes.shape[0]):
            coord[0].append(nodes[i][0])
            coord[1].append(nodes[i][1])
        points = np.asarray(coord)
        contours_pts.append(points)
    return contours_pts

def pointsToBezierCurves(points):
    ## compute each independent edge of the polygon as bezier curve
    edges = []
    a = 1
    b = a + 4
    for p in range(int(points.shape[1]/3)):
        if p == 0:
            a = 1
            b = a + 4
        else:
            a = b - 1
            b = a + 4

        if b > points.shape[1]:
            pts = np.concatenate((points[:,a:points.shape[1]],np.asarray([points[:,0]]).T,np.asarray([points[:,1]]).T),axis=1)
            edges.append(bezier.Curve(pts, degree=3))
        else:
            edges.append(bezier.Curve(points[:,a:b], degree=3))
    return edges

def curvesToCoordinates(edges): ### POTENTIAL BUG points not accessible
    ## compute node coordinates from the bezier curves
    #plt.figure(figsize=(5,5))
    coords = [[],[]]
    sum_pt = 0
    for p in range(int(points.shape[1]/3)):
        nb_points = int(edges[p].length/5)
        sum_pt += nb_points
        coords[0].append(edges[p].evaluate_multi(np.linspace(0,1,nb_points))[0])
        coords[1].append(edges[p].evaluate_multi(np.linspace(0,1,nb_points))[1])

        #plt.plot(edges[p].evaluate_multi(np.linspace(0,1,nb_points))[0].T,edges[p].evaluate_multi(np.linspace(0,1,nb_points))[1].T)
    #plt.axis('equal')

    ## concatenate the coordinates of the edges
    coords_arr = np.zeros((2,sum_pt))
    coords_arr[0] = np.concatenate(np.asarray(coords[0]))
    coords_arr[1] = np.concatenate(np.asarray(coords[1]))

    ## remove last point (first point repeated)
    coords_arr = np.delete(coords_arr,-1,1)

    return coords_arr.T

def computeInnerContour(coords_arr, delta):
  pco = pyclipper.PyclipperOffset()
  pco.AddPath(coords_arr, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

  ## compute "sub-curves"
  tmp = pco.Execute(delta)
  if len(tmp) > 0:
    subcontour = np.asarray(tmp[0])
    return subcontour
  return None

def computeContourLength(data):
    l = 0
    for i in range(data.shape[0]):
        if i == data.shape[0]-1:
            l_edge = math.hypot(data[i][0]-data[0][0],data[i][1]-data[1][0])
        else:
            l_edge = math.hypot(data[i][0]-data[i+1][0],data[i][1]-data[i+1][1])
        l = l + l_edge

    return l

def removeDuplicateVertices(x, y):
    '''Remove duplicate vertices in a contour'''
    min_dist = 1
    dim = x.shape[0]
    newx=[x[0]]
    newy=[y[0]]
    for i in range(1,dim):
        l = np.sqrt(pow(x[i]-newx[-1],2) + pow(y[i]-newy[-1],2))
        if l > min_dist:
            newx.append(x[i])
            newy.append(y[i])
    return np.array(newx), np.array(newy)

def resampleContour(data, interval):
    '''Resample a contour to a homogeneous number of vertices'''
    x, y = removeDuplicateVertices(data[:,0], data[:,1])
    dim = x.shape[0]

    t = [0.0]
    for i in range(1,x.shape[0]+1):
        l = np.sqrt(pow(x[i%dim]-x[(i-1)%dim],2) + pow(y[i%dim]-y[(i-1)%dim],2))
        t.append(t[-1] + l);
    t = np.array(t)
    t = t/t[-1]
    t = np.delete(t, -1)

    # fit polygon
    splx = splrep(t,x)
    sply = splrep(t,y)

    # resample polygon homogeneously
    l = computeContourLength(data)
    num = int(l/interval)
    t = np.linspace(0,1,num)
    t = np.delete(t,-1)
    x1 = splev(t,splx)
    y1 = splev(t,sply)
    #plt.figure(figsize=(30,20))
    #plt.scatter(x1,y1)
    #plt.axis('equal')

    return x1,y1,splx,sply

def plotTwoContours(x1,y1,x2,y2):
    '''Plot two contours, showing their start point and their direction'''
    import matplotlib.pyplot as plt
    plt.scatter(x1,y1)
    plt.scatter(x2,y2)
    plt.scatter(x1[0],y1[0],color='green',s=[200],marker='P')
    plt.scatter(x1[5],y1[5],color='green',s=[200],marker='*')
    plt.scatter(x2[0],y2[0],color='red',s=[200],marker='P')
    plt.scatter(x2[5],y2[5],color='red',s=[200],marker='*')
    plt.axis('equal')

def edgeLength(x, y):
    '''Compute the length of each edge in a contour'''
    import numpy as np
    e = np.array([x-np.roll(x,1), y-np.roll(y,1)]).T
    l = np.sqrt(np.sum(np.power(e,2),axis=1))
    return l

def show_assignments(a, b, P):
    norm_P = P/P.max()
    plt.figure(figsize=(20,15))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            if norm_P[i,j].item()>0.1:
                plt.arrow(a[i, 0], a[i, 1], b[j, 0]-a[i, 0], b[j, 1]-a[i, 1],alpha=norm_P[i,j].item())
    plt.title('Assignments')
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.axis('equal')

def show_assignments_max(a, b, P):
    norm_P = P/P.max()
    plt.figure(figsize=(20,15))
    for i in range(a.shape[0]):
        j=np.argmax(P[i,:])
        plt.arrow(a[i, 0], a[i, 1], b[j, 0]-a[i, 0], b[j, 1]-a[i, 1],alpha=norm_P[i,j].item(),color=(0,1,0))
    for j in range(a.shape[0]):
        i=np.argmax(P[:,j])
        plt.arrow(a[i, 0], a[i, 1], b[j, 0]-a[i, 0], b[j, 1]-a[i, 1],alpha=norm_P[i,j].item(),color=(1,0,0))
    plt.title('Assignments (maximum)')
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.axis('equal')

def show_assignments_proj(a, b, P, show):
    p=np.array(P)
    x=np.array(a)
    y=np.array(b)
    y2=np.diag(1/np.sum(p,axis=1)).dot(p.dot(y))
    if show == True:
        plt.figure(figsize=(20,15))
        for i in range(len(x)):
            plt.arrow(x[i, 0], x[i, 1], y2[i, 0]-x[i, 0], y2[i, 1]-x[i, 1])
        plt.title('Assignments (projection)')
        plt.scatter(x[:, 0], x[:, 1])
        plt.scatter(y2[:, 0], y2[:, 1])
        plt.axis('equal')
    return x,y2

def show_assignments_proj2(a, b, P, show):
    p=np.array(P)
    x=np.array(a)
    y=np.array(b)

    y2 = np.diag(1/np.sum(p,axis=1)).dot(p.dot(y)) # projection
    r = y2-x # projected delta vectors
    rn = np.sqrt(np.sum(r*r,axis=1)) # norm of the deltas

    for i,_ in enumerate(x):
        ni = 0
        s = 0
        for j,_ in enumerate(y):
            if p[i,j] != 0:
                nij = np.sqrt((y[j]-x[i])**2)
                ni += p[i,j]*nij
                s += p[i,j]
        ni /= s
        y2[i] = x[i] + r[i]*ni/rn[i]

    if show == True:
        plt.figure(figsize=(20,15))
        for i in range(x.shape[0]):
            plt.arrow(x[i, 0], x[i, 1], y2[i, 0]-x[i, 0], y2[i, 1]-x[i, 1])
        plt.title('Assignments (projection)')
        plt.scatter(x[:, 0], x[:, 1])
        plt.scatter(y2[:, 0], y2[:, 1])
        plt.axis('equal')
    return x,y2

def show_assignments_proj_maxcoupling(a, b, P, show):
    import matplotlib.pyplot as plt
    import numpy as np
    p=np.array(P)
    x=np.array(a)
    y=np.array(b)
    y2 = np.copy(y)
    if show == True:
        plt.figure(figsize=(20,15))
    for i in range(x.shape[0]):
        if len(np.unique(p[i,:]))>2:
            iy = np.argmax(p[i,:])
            y2[i,:] = y[iy]
        if show == True:
            plt.arrow(x[i, 0], x[i, 1], y2[i, 0]-x[i, 0], y2[i, 1]-x[i, 1])
    if show == True:
        plt.title('Assignments (projection)')
        plt.scatter(x[:, 0], x[:, 1])
        plt.scatter(y2[:, 0], y2[:, 1])
        plt.axis('equal')
    return x,y2

def _lengthparam(line):
    length = []
    prevp = line[0]
    d = 0
    for i,p in enumerate(line):
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

def assignments_proj_line(b, P):
    '''
    weighted average position over the linear ring
    '''
    lb = LinearRing(b)
    lpb = _lengthparam(b)
    res = []
    for i,_ in enumerate(b):
        Q=P[i]
        j = np.argmax(Q)
        sump = np.sum(Q)
        replb = _reparam(lpb, lpb[j])
        replb2 = Q.dot(replb)/sump + lpb[j]
        res.append(replb2)
    y2 = []
    for d in res:
        y2.append(lb.interpolate(d, normalized=False).coords[0])
    return np.array(y2)

def contourEqual(c_p,c_p2):
    ''' test if 2 contours are equal'''
    same = True
    different = 0
    for i in range(c_p.shape[0]):
        if (c_p[i][0]!=c_p2[i][0]) or (c_p[i][1]!=c_p2[i][1]):
            same = False
            different += 1
    #print(same,different)
    return same, different
