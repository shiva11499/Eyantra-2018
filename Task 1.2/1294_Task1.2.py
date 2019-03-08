from aruco_lib import*
import numpy
import math
import csv
import cv2
import cv2.aruco as aruco


def aruco_detect():
    '''
    you will need to modify the ArUco library's API using the dictionary in it to the respective
    one from the list above in the aruco_lib.py. This API's line is the only line of code you are
    allowed to modify in aruco_lib.py!!!
    '''
    img = cv2.imread('Image4.jpg')#give the name of the image with the complete path
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    id_aruco_trace = 0
    det_aruco_list = {}
    img2 = img[0:400,0:400]   #separate out the Aruco image from the whole image
    det_aruco_list = detect_Aruco(img2)
    if det_aruco_list:
        img3 = mark_Aruco(img2,det_aruco_list) 
        id_aruco_trace = calculate_Robot_State(img3,det_aruco_list)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
    parameters = aruco.DetectorParameters_create()
    corners,ids,_= aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
    
    color_detect(img,ids)


def color_detect(img,ids):
    '''
    code for color Image processing to detect the color and shape of the 2 objects at max.
    mentioned in the Task_Description document. Save the resulting images with the shape
    and color detected highlighted by boundary mentioned in the Task_Description document.
    The resulting image should be saved as a jpg. The boundary should be of 25 pixels wide.
    '''
    
    #thresholding the image
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    retval, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    
    #Setting contours for detection of two objects
    cnt = contours[5]
    cnt1 = contours[4]

    #Setting the font
    font = cv2.FONT_HERSHEY_COMPLEX

    #centroid location
    #For 1st object
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print ("Centroid = ", cx, ", ", cy)
    
    #For second object
    M1 = cv2.moments(cnt1)
    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])
    print ("Centroid = ", cx1, ", ", cy1)

    #color detection
    color = img[cy, cx]
    color1 = img[cy1, cx1]
    
    if (color[0] >=0 and color[0]<=10 and color[1] >= 0 and color[1] <= 10 and color[2] >= 250 and color[2] <= 255):
        img= cv2.drawContours(img,[cnt], 0, (0,255,0),25)                                                                                                                       
    elif (color[0] >=110 and color[0]<=255 and color[1] >= 0 and color[1] <= 120 and color[2] >= 0 and color[2] <= 70):                     
        img= cv2.drawContours(img,[cnt], 0, (0,0,255),25)                                                                                                                      
    elif (color[0] >=0 and color[0]<=10 and color[1] >= 120 and color[1] <= 255 and color[2] >= 0 and color[2] <= 70):                      
        img= cv2.drawContours(img,[cnt], 0, (255,0,0),25)

    if (color1[0] >=0 and color1[0]<=10 and color1[1] >= 0 and color1[1] <= 10 and color1[2] >= 250 and color1[2] <= 255):
         img= cv2.drawContours(img,[cnt1], 0, (0,255,0),25)                                                                                                                       
    elif (color1[0] >=110 and color1[0]<=255 and color1[1] >= 0 and color1[1] <= 120 and color1[2] >= 0 and color1[2] <= 70):                     
        img= cv2.drawContours(img,[cnt1], 0, (0,0,255),25)                                                                                                                      
    elif (color1[0] >=0 and color1[0]<=10 and color1[1] >= 120 and color1[1] <= 255 and color1[2] >= 0 and color1[2] <= 70):                      
        img= cv2.drawContours(img,[cnt1], 0, (255,0,0),25)

    cv2.putText(img,"(" + str(cx) + "," + str(cy) + ")",(cx,cy),font,0.6, 0)
    cv2.putText(img,"(" + str(cx1) + "," + str(cy1) + ")",(cx1,cy1),font,0.6, 0)
    
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    epsilon1 = 0.01*cv2.arcLength(cnt1,True)
    approx1 = cv2.approxPolyDP(cnt1,epsilon1,True)


    with open('1294_Task1.12.csv', 'a') as f:
        writer = csv.writer(f)
    
        #writer.writerow(["Image Name","Aruco Id","(x,y) Object1","(x,y) Object2"])
        writer.writerow(["Image5.jpg", ids, "(" + "(" + str (cx)+ ")" + "," + str(cy) + ")", "(" + "(" + str(cx1) + ")" + "," + str(cy1) + ")"])

    cv2.imwrite('Image5.jpg',img)
    cv2.imshow("ColorImage",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    aruco_detect()

