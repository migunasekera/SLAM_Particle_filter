#Daniel D. Lee, Alex Kushleyev, Kelsey Saulnier, Nikolay Atanasov
cimport cython
import numpy as np

# INPUT 
# im              the map 
# x_im,y_im       physical x,y positions of the grid map cells
# vp(0:2,:)       occupied x,y positions from range sensor (in physical unit)  
# xs,ys           physical x,y,positions you want to evaluate "correlation" 
#
# OUTPUT 
# c               sum of the cell values of all the positions hit by range sensor
def mapCorrelation_fclad(im, x_im, y_im, vp, xs, ys):
  cdef double nx = im.shape[0]
  cdef double ny = im.shape[1]
  cdef double xmin = x_im[0]
  cdef double xmax = x_im[-1]
  cdef double xresolution = (xmax-xmin)/(nx-1)
  cdef double ymin = y_im[0]
  cdef double ymax = y_im[-1]
  cdef double yresolution = (ymax-ymin)/(ny-1)
  cdef int nxs = xs.size
  cdef int nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                              np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


#Bresenham's line algorithm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def getMapCellsFromRay_fclad(int x0t, int y0t,
                             xis,yis, 
                             int maxMap):
        cdef int nPoints, index
        cdef int x, y
        cdef int x_iterator
        cdef int x1, y1, temp
        cdef int x0, y0
        cdef float error
        cdef int deltax, deltay
        cdef int ystep
        cdef int steep
        nPoints = np.size(xis)
        index = 0
        lineMap = np.zeros([maxMap * nPoints, 2], dtype=np.int16)
        cdef short [:, :] lineMap_view = lineMap
        cdef short [:] xis_view = xis
        cdef short [:] yis_view = yis
        for idx in range(nPoints):
                x1 = xis_view[idx]
                y1 = yis_view[idx]
                x0 = x0t
                y0 = y0t
                steep = (abs(y1-y0) > abs(x1-x0))
                if steep:
                        temp = x0
                        x0 = y0
                        y0 = temp
                        temp = x1
                        x1 = y1
                        y1 = temp
                if x0 > x1:
                        temp = x0
                        x0 = x1
                        x1 = temp
                        temp = y0
                        y0 = y1
                        y1 = temp
                deltax = x1 - x0
                deltay = abs(y1 - y0)
                error = deltax / 2.
                y = y0
                ystep = 0
                if y0 < y1:
                  ystep = 1
                else:
                  ystep = -1
                x_iterator = x0
                if steep:
                        while(x_iterator < x1):
                                lineMap_view[index, 0] = y
                                lineMap_view[index, 1] = x_iterator
                                index += 1
                                error = error - deltay
                                if error < 0:
                                        y += ystep
                                        error += deltax
                                x_iterator += 1
                else:
                        while(x_iterator < x1):
                                lineMap_view[index, 0] = x_iterator
                                lineMap_view[index, 1] = y
                                index += 1
                                error = error - deltay
                                if error < 0:
                                        y += ystep
                                        error += deltax
                                x_iterator += 1
        xyio = lineMap[:index, :].T
        return xyio
