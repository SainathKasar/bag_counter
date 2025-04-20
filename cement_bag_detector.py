import cv2 as cv
import numpy as np

video=cv.VideoCapture("F://Videos//20220710_121852.mp4") #your video file
fix_frame=300
#properties of rectangle
min_width=800
min_height=800
algo=cv.createBackgroundSubtractorMOG2()

'''def center_Pthandle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    center_x=x+x1
    center_x=y+y1
    return center_x,center_y
detect=[]
offsect=5 #acts like relative error in pixel'''

while True:
    ret,frame=video.read()
    cv.imshow("initial_image",frame)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(frame,(3,3),5)
    img_applied=algo.apply(blur)
    cv.waitKey(10)
#special function for each frame:
    img_applied=algo.apply(blur)
    dilate_img=cv.dilate(img_applied,np.ones((5,5)))
    box_kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    get_INNER_part=cv.morphologyEx(dilate_img,cv.MORPH_CLOSE,box_kernel)
    get_INNER_Part=cv.morphologyEx(get_INNER_part,cv.MORPH_CLOSE,box_kernel)
    countour_shape=cv.findContours(get_INNER_Part,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    #cv.line(frame,(320,0),(320,480),(155,155,155),3)
#to make box to cement bag
    for (i,c) in enumerate(countour_shape):
        (x,y,w,h)=cv.boundingRect(c)
        validate_counter=(w<=min_width)and(h<=min_height) #properties of bag
        if not validate_counter:
            continue
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    center_point=center_Pthandle(x,y,w,h)
    detect.append(center)
    cv.circle(frame,center,4,(0,0,255),-1) #point on center of image

    for(x,y) in detect:
        if y<(fix_frame+offset) and y>(fix_frame-offset):
            counter+=1
        detect.remove(x,y)

    #cv.putText(frame,"Cement Bags"+str(counter),(0,10),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    
    
cv.release()
cv.destroyAllWindows()
