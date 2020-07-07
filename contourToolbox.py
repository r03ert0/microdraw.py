import numpy as np

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


def resampleContour(data, num):
    '''Resample a contour to a homogeneous number of vertices'''
    from scipy.interpolate import splev, splrep

    x, y = removeDuplicateVertices(data[:,0], data[:,1])
    dim = x.shape[0]
    #print('num',num)
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
    t = np.linspace(0,1,num)
    t = np.delete(t,-1)
    #print(t)
    x1 = splev(t,splx)
    y1 = splev(t,sply)
    #plt.plot(x1,y1)
    #plt.axis('equal')

    return x1,y1,splx,sply

def extractMainContours(img,hsl, thres, delta, dens, path, name):
    """
    * img: image from which the contour will be extracted
    * delta: for the widening of the contour with Pyclipper
    * dens: density at which the contour will be resampled
    * path: where the intermediate plots will be saved
    * name: basename for the figures
    """

    import cv2
    import matplotlib.pyplot as plt
    import scipy.ndimage.morphology as morph
    from skimage import measure
    import pyclipper

    ## identify the segmentation that are potentially spurious
    missedSeg = 0

    ## convert img to HLS format
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    fig,ax = plt.subplots(2,3,figsize=(10,7))
    fig.suptitle(name)
    ## 1 - threshold brain
    brain = np.zeros(img_hsl[:,:,0].shape, dtype=np.uint8)
    brain[img_hsl[:,:,hsl]<thres] = 1
    im = ax[0,0].imshow(brain)
    ax[0,0].set_title('1- Image thresholded')
    #plt.colorbar(im, ax = ax[0,0])
    #plt.savefig(path+name[:-4]+'_1thres.png')
    #plt.close()

    ## 2 - use morphological transformations to get a compact silhouette of the
    ## brain while keeping as much as possible the delineation of the sulci
    brain1 = morph.binary_closing(brain,iterations=5)
    brain1 = morph.binary_opening(brain1,iterations=10)
    brain1 = morph.binary_closing(brain1,iterations=5)
    #plt.figure()
    ax[0,1].imshow(brain1)
    ax[0,1].set_title('2- After morph transfo')
    #plt.savefig(path+name[:-4]+'_2thres_opening.png')
    #plt.close()

    ## 3 - find contours
    contours = measure.find_contours(brain1,0.90,'high')
    ## sort by contour length
    contours_sorted = sorted(contours,key=len)
    C = []
    #plt.figure()
    ax[1,0].imshow(brain, cmap=plt.cm.gray)
    for i in range(0,len(contours_sorted),1):
        if len(contours_sorted[len(contours_sorted)-i-1]) > len(contours_sorted[-1])/4:
            C.append(contours_sorted[len(contours_sorted)-i-1])
        else:
            break
    for c in C:
        ax[1,0].plot(c[:,1],c[:,0],linewidth=3)
    ax[1,0].set_title('3- Selected contours')
    #plt.savefig(path+name[:-4]+'_3rawseg.png')
    #plt.close()

    ## 4 - widen and simplify the contour
    C_clip = []
    for c in C:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(c, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        ## compute "sub-curves"
        C_clip.append(np.asarray(pco.Execute(delta)[0]))
    #plt.figure()
    ax[1,1].imshow(img, cmap=plt.cm.gray)
    for c in C_clip:
        ax[1,1].scatter(c[:,1],c[:,0],linewidth=2,marker='.')
    ax[1,1].set_title('4- Widen & simplified')
    #plt.savefig(path+name[:-4]+'_4seg_pyclipper.png')
    #plt.close()

    ## 5 - resample the contour with a homogeneous density of nodes
    C_resampled = []
    for c in C_clip:
        L = []
        for i in range(len(c)):
            L.append(np.linalg.norm(c[i]-c[(i+1)%len(c)]))
        n = sum(L)*dens
        #print(n)
        if n >= 2:
            C_resampled.append(resampleContour(c, n))
        #print(sum(L))
    #plt.figure()
    ax[1,2].imshow(img, cmap=plt.cm.gray)
    for c in C_resampled:
        ax[1,2].scatter(c[1], c[0], linewidth=2, marker='.')
    ax[1,2].set_title('5- Resampled')
    #plt.savefig(path+name[:-4]+'_5seg_pyclipper_resampled.png')

    ## check missed segmentations
    if len(C)>0:
        L3 = []
        for i in range(len(C[0])):
            L3.append(np.linalg.norm(C[0][i]-C[0][(i+1)%len(C[0])]))
        L4 = []
        for i in range(len(C_clip[0])):
            L4.append(np.linalg.norm(C_clip[0][i]-C_clip[0][(i+1)%len(C_clip[0])]))
        if sum(L4) < sum(L3)/4:
            missedSeg = 1

    ax[0,2].axis('off')
    plt.tight_layout()
    plt.savefig(path+name+'_autocontours.png')
    plt.close()

    return C_resampled, missedSeg

def saveContoursAsJson(subject,slicenb,contours,savepath):

    from string import Template
    import json

    ## create template to be filled with the data
    jsontemplate = Template('{"fileID":"/${subject}/${subject}.json&slice=${slicenb}","user":"celinede",${annotations}}')
    annotationtemplate = Template('"annotation":{"path":["Path",{"applyMatrix":true,"segments":[${segments}],"closed":true,"fillcolor":[0.69412,0.72941,0.56078,0.5],"strokeColor":[0,0,0],"strokeScaling":false}],"name":"${contourname}"}')
    segtemplate = Template('[${x},${y}]')
    #segtemplate = Template('{"x":${x},"y":${y},"selected":true}') ### if annotation selected

    annotationset = ''
    cnb = 0
    #if len(contours)>0:
    for cc in contours: ## for each contour of the given image
        segments = ''
        cnb = cnb+1
        for p in range(len(cc[0])): ## for each point defining the contour
            if (cc[0][p] >= 0) and (cc[1][p] >= 0):
                pt = segtemplate.substitute(x=cc[0][p],y=cc[1][p])
                if segments == '':
                    segments = pt
                else:
                    segments = segments+','+pt
        annotation = annotationtemplate.substitute(segments=segments,contourname='CONTOUR'+str(cnb))
        jsondata = jsontemplate.substitute(subject=subject,slicenb=slicenb,annotations=annotation)

        if annotationset == '':
            annotationset = jsondata
        else:
            annotationset = annotationset+','+jsondata


    data = json.loads('['+annotationset+']')
    with open(savepath+subject+'_'+"{0:0=3d}".format(slicenb)+'.json','w') as fp:
        json.dump(data,fp)

def saveContoursAsJsonTo1000(subject,slicenb,contours,imgwidth,savepath):

    from string import Template
    import json

    ## create template to be filled with the data
    jsontemplate = Template('{"fileID":"/${subject}/${subject}.json&slice=${slicenb}","user":"celinede",${annotations}}')
    annotationtemplate = Template('"annotation":{"path":["Path",{"applyMatrix":true,"segments":[${segments}],"closed":true,"fillcolor":[0.69412,0.72941,0.56078,0.5],"strokeColor":[0,0,0],"strokeScaling":false}],"name":"${contourname}"}')
    segtemplate = Template('[${x},${y}]')
    #segtemplate = Template('{"x":${x},"y":${y},"selected":true}') ### if annotation selected

    annotationset = ''
    cnb = 0
    #if len(contours)>0:
    for cc in contours: ## for each contour of the given image
        segments = ''
        cnb = cnb+1
        for p in range(len(cc[0])): ## for each point defining the contour
            if (cc[0][p] >= 0) and (cc[1][p] >= 0):
                pt = segtemplate.substitute(x=cc[1][p]*(1000/imgwidth),y=cc[0][p]*(1000/imgwidth)) ## scaling for microdraw
                if segments == '':
                    segments = pt
                else:
                    segments = segments+','+pt
        annotation = annotationtemplate.substitute(segments=segments,contourname='CONTOUR'+str(cnb))
        jsondata = jsontemplate.substitute(subject=subject,slicenb=slicenb,annotations=annotation)

        if annotationset == '':
            annotationset = jsondata
        else:
            annotationset = annotationset+','+jsondata


    data = json.loads('['+annotationset+']')
    with open(savepath+subject+'_'+"{0:0=3d}".format(slicenb)+'.json','w') as fp:
        json.dump(data,fp)


def extractNodesFromJsonPolygon(jsonPath):
    import json
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

def importContoursFromJsons(path,subject):
    '''
    imports list of contours from a series of json files for one subject
    :param path: path to the folder containing the json files
    :param subject: name of the subject to be imported (file names should start with the subject identification)
    :return c_pts: list of contours
    '''
    import os
    files = os.listdir(path)
    c_pts = []
    for f in sorted(files):
        if f.startswith(subject) & f.endswith('.json'):
            print(f)
            c_pts.append(extractNodesFromJsonPolygon(path+f))
    return c_pts
