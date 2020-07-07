def dx(distance, m):
    import math
    return math.sqrt(distance/(m**2+1))

def dy(distance, m):
    return m*dx(distance, m)

def computePointOnHalfLine(a, b, distance):
    import math
    #### compute line parameters from the 2 matching points of the contours (y=mx+p)
    m = (a[1]-b[1])/(a[0]-b[0])
    p = a[1]-(a[0]*m)

    point_a = a
    point_b = (point_a[0]+dx(distance,m), point_a[1]+dy(distance,m))
    other_possible_point_b = (point_a[0]-dx(distance,m), point_a[1]-dy(distance,m)) # going the other way

    # decifer which point of the solution is the valid one
    if math.sqrt((point_b[0]-b[0])**2+(point_b[1]-b[1])**2)>math.sqrt((other_possible_point_b[0]-b[0])**2+(other_possible_point_b[1]-b[1])**2):
        b_dist = other_possible_point_b
    else:
        b_dist = point_b
    return b_dist

def rollingMean(x, w):
    import numpy as np
    if w % 2 != 0:
        print('incompatible odd width')
        return 0
    else:
        res = np.zeros((len(x),2))
        x_roll = np.roll(x,int(w/2))
        for i in range(len(x)):
            x_roll_tmp = np.copy(x_roll)
            x_roll = np.roll(x_roll_tmp,-1)
            #print(x_roll)
            res[i,:] = np.asarray([np.mean(x_roll[0:w]),np.std(x_roll[0:w])])
            #print('i '+str(i))
            #print(x_roll[0:w])
            #print('res=',res[i])
        return res
