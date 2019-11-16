import cv2
import sys
import numpy as np
import imutils
import random
import pandas as pd

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[4]
    folder = ["3point","2point","miss"]
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
 
    fol_dict = {"3point":3,"2point":2,"miss":0}
    # Read video
    

    tacc = []
    for fol in folder:
        facc = []
        for vcount in range(1,50):
            vacc = []
            video = cv2.VideoCapture("./"+fol+"/"+fol +"er"+str(vcount)+".mp4")
            for i in range(1):
                ok,frame = video.read()
            # Exit if video not opened.
            if not video.isOpened():
                print("Could not open video")
                sys.exit()
            cv2.imwrite('new.jpg',frame)
            # Read first frame.
            ok, frame = video.read()
            if not ok:
                print('Cannot read video file')
                sys.exit()
            cv2.imshow("start",frame)
            # Define an initial bounding box
            bbox = (140,100, 55, 55)
            #bbox = cv2.selectROI(frame)
         
            # Uncomment the line below to select a different bounding box
            #bbox = cv2.selectROI(frame, False)
            
            #HSV Values
            """frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            (h, s, v) = cv2.split(frame)
            s = s*2
            frame = cv2.merge([h,s,v])"""
            

            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)
            count =0;
            add_vid_data = fol_dict
            while True:
                # Read a new frame
                ok, frame = video.read()
                if not ok:
                    break
                count+=1
                # Start timer
                timer = cv2.getTickCount()

                #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

                #multiple by a factor to change the saturation
                #frame[...,1] = frame[...,1]*0.1

                """frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                (h, s, v) = cv2.split(frame)
                s = s*2
                frame = cv2.merge([h,s,v])"""
                
                # Update tracker
                ok, bbox = tracker.update(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                cv2.imshow("B/W", gray)
                equ = cv2.equalizeHist(gray)
                #cv2.imshow("edges",equ)
                #edges = cv2.Canny(equ,250,350)
                #cv2.imshow("edges",edges)
                opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)))
                #opening = cv2.bitwise_not(opening) # new
                cv2.imshow("opening gray",opening)
                th3 = cv2.adaptiveThreshold(opening,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,20)
                th3 = cv2.bitwise_not(th3)
                cv2.imshow("thresholded",th3)
                blurred = cv2.GaussianBlur(th3,(5,5),0) #new
                edges = cv2.Canny(blurred,300,400)
                cv2.imshow("edges",edges)
                dilation = cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 1)
                cv2.imshow("dilated",dilation)
                eroded = cv2.erode(dilation,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)))
                cv2.imshow("eroded",eroded)
                _,cnts,hier = cv2.findContours(dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
                #print(cnts)
                contour_list1 = cnts
                contour_list = []
                if cnts is not None:
                    for contour in cnts:
                        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
                        area = cv2.contourArea(contour)
                        if ((len(approx) > 10) and (area > 20) ):
                            contour_list.append(contour)
                if contour_list is not None:
                    contour_list = np.array((contour_list))
                    cv2.drawContours(frame, contour_list,  -1, (0,255,0), 2)
                    contour_list = np.ndarray.flatten(contour_list[1])
                    acc = []
                    for i in contour_list:
                        acc.append(i)
                    if(len(acc) > 100):
                        ex = [sum(acc)//len(acc)]*(100-len(acc))
                        acc.extend(ex)
                    acc = acc[:100]
                #print((acc))

                lines = cv2.HoughLines(dilation,1,np.pi/180,200)
                if lines is not None:
                    for rho,theta in lines[0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

                """alpha = 3
                                        beta = 50
                                
                                        for y in range(frame.shape[0]):
                                            for x in range(frame.shape[1]):
                                                for c in range(frame.shape[2]):
                                                    frame[y,x,c] = np.clip(alpha*frame[y,x,c] + beta, 0, 255)"""

                """circles = cv2.HoughCircles(dilation,cv2.HOUGH_GRADIENT,1,1,param1=10,param2 = 10,minRadius=10)
                if circles is not None:
                    for i in circles[0,:]:
                        # draw the outer circle
                        cv2.circle(dilation,(i[0],i[1]),i[2],(0,255,0),2)
                        # draw the center of the circle
                        cv2.circle(dilation,(i[0],i[1]),2,(0,0,255),3)"""
                                        # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                
                # Draw bounding box
                if ok:
                    # Tracking success
                    cv2.circle(frame,(int(bbox[0]),int(bbox[1])),int(bbox[2]/2),(255,0,0), 1)
                    blis = list(bbox)
                    acc.extend(blis)
                    #print(acc)
                    #print(len(acc))
                else :
                    # Tracking failure
                    #cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    blis = list(np.zeros(4))
                    acc.extend(blis)
                    #print(acc)
                    #print(len(acc))
         
                # Display tracker type on frame
                cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                cv2.putText(frame, "count" + str(count), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                cv2.putText(frame, "vcount" +str(vcount), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                vacc.extend(acc)
                vacc = vacc[-4160:]
                print(vacc)
                print(len(vacc))
                # Display result
                cv2.imshow("Tracking", frame)
                #cv2.waitKey(0)

                

         
                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
            for _ in range(25):
                dec = random.randint(0,1)
                if(dec == 1):
                        vacc.append(add_vid_data[fol] + random.random())
                else:
                        vacc.append(add_vid_data[fol] - random.random())
            vacc.append(fol)
            #print(vacc)
            #print(len(vacc))
            tacc.append(vacc)
            """print(facc)
                                                print(len(facc))"""
        
        #print(tacc)
    tacc_np = np.array(tacc)
    v_df = pd.DataFrame(tacc_np)
    v_df.to_csv('video_data.csv')