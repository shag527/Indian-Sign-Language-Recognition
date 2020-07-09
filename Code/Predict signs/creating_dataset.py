
def cd_main():
    # importing necessary files
    import cv2
    import imutils
    import numpy as np
    import os
    from os import path
    import tkinter as tk
    from tkinter import  messagebox

    bg = None

    #To find the running average over the background
    def run_avg(image,aweight):
        nonlocal bg
        #initialize the background
        if bg is None:
            bg=image.copy().astype("float")
            return
        cv2.accumulateWeighted(image,bg,aweight)

    # Segment the egion of hand
    def extract_hand(image,threshold=25):
        nonlocal bg
        diff=cv2.absdiff(bg.astype("uint8"),image)
        thresh=cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
        (_,cnts,_)=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if(len(cnts)==0):
            return
        else:
            max_cont=max(cnts,key=cv2.contourArea)
            return (thresh,max_cont)

    def n(x):
        pass


    aWeight=0.5
    cam=cv2.VideoCapture(0)
    #t,r,b,l=100,350,228,478
    t,r,b,l=100,350,325,575
    num_frames=0

    cur_mode=None
    count=0
    limit=500
    method=1

    #print("Enter the method: (1 for keeping bg still and 2 for skin extraction")
    #method=int(input())
    option=messagebox.askquestion('Select option','Choose default method ?')
    if option=='yes':
        method=2
    else:
        method=1

    if method==2:
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 640, 480)
        cv2.createTrackbar("LH", "Tracking", 0, 255, n)
        cv2.createTrackbar("LS", "Tracking", 0, 255, n)
        cv2.createTrackbar("LV", "Tracking", 0, 255, n)
        cv2.createTrackbar("UH", "Tracking", 255, 255, n)
        cv2.createTrackbar("US", "Tracking", 32, 255, n)
        cv2.createTrackbar("UV", "Tracking", 255, 255, n)

    while(cam.isOpened()):
        _,frame=cam.read()
        if frame is not None:
            frame=imutils.resize(frame,width=700)
            frame=cv2.flip(frame,1)
            clone=frame.copy()

            # height,width=frame.shape[:2]
            roi=frame[t:b,r:l]

            if method ==1:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                if(num_frames<30):
                    run_avg(gray,aWeight)
                    cv2.putText(clone, "Keep the Camera still.", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
                else:
                    cv2.putText(clone, "Press esc to exit.", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(clone, "Keep the Camera still.", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
                    cv2.putText(clone, "Put your hand in the rectangle", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(clone, "Press the key of the sample", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                    hand=extract_hand(gray)
                    if hand is not None:
                        thresh, max_cont = hand
                        mask = cv2.drawContours(clone, [max_cont + (r, t)], -1, (0, 0, 255))
                        cv2.imshow("Threshold", thresh)
                        mask = np.zeros(thresh.shape, dtype="uint8")
                        cv2.drawContours(mask, [max_cont], -1, 255, -1)
                        mask = cv2.medianBlur(mask, 5)
                        mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        res = cv2.bitwise_and(roi, roi, mask=mask)
                        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                        cv2.imshow("Extracted", res)
                        high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        lowThresh = 0.5 * high_thresh
                        res = cv2.Canny(res, lowThresh, high_thresh)

            if method==2:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lh = cv2.getTrackbarPos("LH", "Tracking")
                ls = cv2.getTrackbarPos("LS", "Tracking")
                lv = cv2.getTrackbarPos("LV", "Tracking")
                uh = cv2.getTrackbarPos("UH", "Tracking")
                us = cv2.getTrackbarPos("US", "Tracking")
                uv = cv2.getTrackbarPos("UV", "Tracking")

                l_b = np.array([lh, ls, lv])
                u_b = np.array([uh, us, uv])

                cv2.putText(clone, "Put your hand in the rectangle", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0))
                cv2.putText(clone, "Adjust the values using trackbar", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                cv2.putText(clone, "Press the key of the sample", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                cv2.putText(clone, "Press esc to exit.", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                mask = cv2.inRange(hsv, l_b, u_b)
                cv2.imshow('mask', mask)
                mask = cv2.bitwise_not(mask)
                mask = cv2.medianBlur(mask, 5)
                mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                res = cv2.bitwise_and(roi, roi, mask=mask)
                res=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
                cv2.imshow('res', res)
                high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                lowThresh = 0.5 * high_thresh
                res = cv2.Canny(res, lowThresh, high_thresh)



            # hand = cv2.bitwise_and(gray, gray, mask=thresh)
            # cv2.imshow("Hand", hand)
            # res = cv2.Canny(hand, lowThresh, high_thresh)
            # cv2.imshow("Hand res", res)
            # v = np.median(res)
            # sigma=0.33


            if cur_mode!=-1 and cur_mode!=255 and cur_mode is not None:
                file_path = 'Saved Dataset\\'+str(chr(cur_mode))
                if not path.exists(file_path):
                    os.makedirs(file_path)
                if(count<=limit):
                    cv2.imwrite(file_path+'\\'+str(count)+'.jpg',res)
                    print(count)
                    if(count==limit):
                        print("Completed")
                count+=1

            cv2.rectangle(clone,(l,t),(r,b),(0,255,0),2)
            num_frames+=1
            cv2.imshow("Video Feed",clone)
        else:
            messagebox.showerror("error","Can't grab frame")
            break
        k=cv2.waitKey(1)& 0xFF
        if (k==27):
           break
        if(k!=-1 and k!=255 and k!=cur_mode):
            cur_mode=k
            count = 0

    cam.release()
    cv2.destroyAllWindows()

#cd_main()