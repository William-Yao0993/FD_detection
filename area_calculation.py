import cv2 
import numpy as np
import warnings
from config import RATIO_CANOLA,SCALEBAR_UNIT
import os



#path ='d:\OneDrive - Australian National University\Stomata project\graphs'
#------------------------------------------------------------------------------------------------------------------
# Verison 1.1 
# Mask calculator: return Mask area with mm2, speeding up with torch calculation 
#line detect -> add angle filter 
#------------------------------------------------------------------------------------------------------------------
def img_area(img,ratio):
     if isinstance(img,str):
          img=cv2.imread(img,cv2.IMREAD_COLOR)
     x_in_pxl, y_in_pxl = img.shape[:2]
     x_in_mm = x_in_pxl * ratio
     y_in_mm = y_in_pxl * ratio
     area_in_mm2= x_in_mm * y_in_mm
     return area_in_mm2
def line_detect(img):
     '''
        Detect and filter the best candidate for scale bar 
        Return:
        if not find -> None 
        success ->  Lines ndarray
     '''
     if isinstance(img,str):
        if os.path.isfile(img):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
     x_start, x_end = int(img.shape[0] * 0.0), int(img.shape[0] * 0.15)  
     y_start, y_end = int(img.shape[1] * 0.60), int(img.shape[1] * 0.73)

     img_roi = img[y_start:y_end,x_start:x_end]
     try:
          img_roi = cv2.GaussianBlur(img_roi,(3,3),0)
          
          low_threshold = 45
          high_threahold = int (2.5 * low_threshold)
          edges = cv2.Canny(img_roi, low_threshold,high_threahold)
          #cv2.imshow('edge',edges)
          rho= 1
          theta= np.pi/180
          threshold= 15
          minLineLength=100
          maxLineGap= 30
          min_theta=0
          max_theta=np.pi/2 

          lines = cv2.HoughLinesP(
               edges,
               rho=rho,
               theta=theta,
               threshold=threshold,
               minLineLength=minLineLength,
               maxLineGap=maxLineGap)
          
          if lines is None:
               warnings.warn(f'Fail to detect scale bar, using default ratio: {RATIO_CANOLA}')
               return RATIO_CANOLA

          # im_cp = img_roi.copy()
          # for x0,y0,x1,y1 in lines.squeeze():
          #      cv2.line(im_cp,(x0,y0),(x1,y1),(0,0,255),3,0)
          #cv2.imshow('  ',im_cp)        

          # Angle filter
          angles = np.arctan2(np.abs(lines[...,1]-lines[...,3]),np.abs(lines[...,0]-lines[...,2]))
          degrees = angles*180/np.pi
          threshold = 2
          angle_mask = (
               (degrees <=threshold) |
               (np.abs(degrees -90)<= threshold)
          )

          lines = lines[angle_mask]
          x_differences = np.abs(lines[...,0] - lines[...,2])
          y_differences = np.abs(lines[...,1] - lines[...,3])
          x_candidate = np.max(x_differences)
          y_candidate = np.max(y_differences)
          x = lines[x_differences== x_candidate].squeeze()
          y = lines[y_differences== y_candidate].squeeze()

          # cv2.line(img_roi,(x[0],x[1]),(x[2],x[3]),(0,0,255),3,0)
          # cv2.line(img_roi,(y[0],y[1]),(y[2],y[3]),(0,0,255),3,0)
          # cv2.imshow('result',img_roi)
          # cv2.waitKey()
          # cv2.destroyAllWindows()
          print(max(x_candidate,y_candidate))
          ratio_to_mm = SCALEBAR_UNIT/max(x_candidate,y_candidate)
          #print(ratio_to_mm)
          if np.abs(ratio_to_mm-RATIO_CANOLA)>1E-3:
               warnings.warn(f'Detected Ratio is much different with default value, using default ratio: {RATIO_CANOLA}')
               return RATIO_CANOLA
     except Exception as e :
          print(e)
          warnings.warn(f'Fail to detect scale bar, using default ratio: {RATIO_CANOLA}')
          return RATIO_CANOLA
     return ratio_to_mm

def masks_calculator(masks,ratio_to_mm,shape=None):
    '''
        Calculate the Area of each Mask in mm2
        masks -> Results Class (data,orig_shape)
        masks.data -> Torch.tensor with shape N x H x W (H and W could differ with original size)
        Return:
        ndarray with shape N, e.g.   ndarray([s1_in_mm2,s2_in_mm2,...])
    '''
    if masks is None:
         return np.zeros(1) # No Masks Given -> size is 0 
    elif shape is not None:
         orig_x,orig_y = shape
    else:
         orig_x, orig_y = masks.orig_shape
    ratio_x,ratio_y = orig_x/masks.data.shape[1],orig_y/masks.data.shape[2] 
    # print(ratio_x,ratio_y)
    areas_in_pxl = masks.data.numpy().sum(axis=(1,2))
    # print(np.count_nonzero(masks.data,axis=(1,2)))
    areas_in_pxl_orig_shape = areas_in_pxl * ratio_x * ratio_y
    
    # convert areas from original shape pixel to mm2 
    return areas_in_pxl_orig_shape * (ratio_to_mm**2)

# if __name__ == '__main__':
#     img= r'c:\Users\u6771897\Desktop\bottom\2024-02-12 D8 P1 B-T\04.jpg'
#     ratio = line_detect(img)
#     print(ratio)
#     print(img_area(img, ratio))