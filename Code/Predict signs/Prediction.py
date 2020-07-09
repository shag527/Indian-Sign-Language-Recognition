def pred_main():
    # importing necessary libraries
    import cv2
    import imutils
    import numpy as np
    import os
    from os import path
    import pickle
    import imageio
    from scipy import ndimage
    from scipy.spatial import distance
    import pyttsx3
    import tensorflow as tf
    from tensorflow import keras
    import keras
    from threading import Thread
    from tkinter import messagebox
    import tkinter as tk


    #global variables
    bg=None
    visual_dict={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',
             28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}
    aWeight=0.5
    cam=cv2.VideoCapture(0)
    #t,r,b,l=100,350,228,478

    # Global Variables
    t,r,b,l=100,350,325,575
    num_frames=0
    cur_mode=None
    predict_sign=None
    count=0
    shape=180
    result_list=[]
    words_list=[]
    prev_sign=None
    count_same_sign=0

    method = 1

    option = messagebox.askquestion('Select option', 'Choose default method ?')
    if option == 'yes':
        method = 2
    else:
        method = 1

    model='files/CNN'

    infile = open(model,'rb')
    cnn = pickle.load(infile)
    infile.close()

    bg=None
    count=0

    #To find the running average over the background
    def run_avg(image,aweight):
        nonlocal bg #initialize the background
        if bg is None:
            bg=image.copy().astype("float")
            return
        cv2.accumulateWeighted(image,bg,aweight)

    #Segment the egion of hand
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

    # output the recognized sign in form of speech
    engine=pyttsx3.init()
    engine.setProperty("rate",100)
    voices=engine.getProperty("voices")
    engine.setProperty("voice",voices[1].id)

    def say_sign(sign):
        while engine._inLoop:
            pass
        engine.say(sign)
        engine.runAndWait()

    def n(x):
        pass

    if method == 2:
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 640, 480)
        cv2.createTrackbar("LH", "Tracking", 0, 255, n)
        cv2.createTrackbar("LS", "Tracking", 0, 255, n)
        cv2.createTrackbar("LV", "Tracking", 0, 255, n)
        cv2.createTrackbar("UH", "Tracking", 255, 255, n)
        cv2.createTrackbar("US", "Tracking", 32, 255, n)
        cv2.createTrackbar("UV", "Tracking", 255, 255, n)

    while(cam.isOpened()):
        _,frame=cam.read(cv2.CAP_DSHOW)
        if frame is not None:
            orig_signs=cv2.imread('files/signs.png')
            signs=cv2.resize(orig_signs,(600,600))
            cv2.imshow("Signs",signs)
            frame=imutils.resize(frame,width=700)
            frame=cv2.flip(frame,1)
            clone=frame.copy()

            # height,width=frame.shape[:2]
            roi=frame[t:b,r:l]

            if method==1:
                gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                gray=cv2.GaussianBlur(gray,(7,7),0)

                if(num_frames<30):
                    run_avg(gray,aWeight)
                    cv2.putText(clone, "Keep the Camera still.", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
                else:
                    cv2.putText(clone, "Press esc to exit.", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(clone, "Keep the Camera still.", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0))
                    cv2.putText(clone, "Put your hand in the rectangle", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0))
                    cv2.putText(clone, "Press the key of the sample", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0))
                    hand=extract_hand(gray)
                    if hand is not None:
                        thresh,max_cont=hand
                        mask=cv2.drawContours(clone,[max_cont+(r,t)],-1, (0, 0, 255))
                        cv2.imshow("Threshold",thresh)
                        mask=np.zeros(thresh.shape,dtype="uint8")
                        cv2.drawContours(mask,[max_cont],-1,255,-1)
                        mask = cv2.medianBlur(mask, 5)
                        mask = cv2.addWeighted(mask, 0.5, mask, 0.5, 0.0)
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        res=cv2.bitwise_and(roi,roi,mask=mask)
                        res=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

                        #---- Apply automatic Canny edge detection using the computed median----

                        high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        lowThresh = 0.5 * high_thresh
                        #res=cv2.Canny(res,lowThresh,high_thresh)

                        #cv2.imshow("Segmented",res)
                        hand=cv2.bitwise_and(gray,gray,mask=thresh)
                        cv2.imshow("Hand",hand)
                        res = cv2.Canny(hand, lowThresh, high_thresh)

                        # Bag of Visual Words
                        '''surf = cv2.xfeatures2d.SURF_create()
                        kp, desc = surf.detectAndCompute(res, None)
                        #print("Surf features extracted!")
                        features = cv2.drawKeypoints(res, kp, None)
                        cv2.imshow("Surf Features",features)

                        if desc is not None:
                            visual_words=kmeans.predict(desc)
                            hist = np.array(np.bincount(visual_words,minlength=shape))
                            hist=hist.reshape(1,-1)
                            sign = svm.predict(hist)
                            # output=visual_dict[sign[0]]
                            cv2.putText(clone,output, (10, 300), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))'''

                        # CNN Model
                        if res is not None and cv2.contourArea(max_cont) > 1000:
                            final_res = cv2.resize(res, (100, 100))
                            final_res = np.array(final_res)
                            final_res = final_res.reshape((-1, 100, 100, 1))
                            final_res.astype('float32')
                            final_res = final_res / 255.0
                            output = cnn.predict(final_res)
                            prob = np.amax(output)
                            sign = np.argmax(output)
                            final_sign = visual_dict[sign]
                            cv2.putText(clone, 'Sign ' + str(final_sign), (10, 200), cv2.FONT_HERSHEY_COMPLEX, 2,
                                        (0, 0, 255))

                            # print(count)
                            count += 1
                            if (count > 10 and count <= 50):
                                if (prob * 100 > 95):
                                    result_list.append(final_sign)
                                    # print(sign, prob)
                            elif (count > 50):
                                count = 0
                                if len(result_list):
                                    predict_sign = (max(set(result_list), key=result_list.count))
                                    result_list = []
                                    if prev_sign is not None:
                                        if prev_sign != predict_sign:
                                            #print(words_list)
                                            words_list += str(predict_sign)
                                            Thread(target=say_sign, args=(predict_sign,)).start()
                                    else:
                                        Thread(target=say_sign, args=(predict_sign,)).start()
                                        # prev_sign=predict_sign
                                    prev_sign = predict_sign
                                # print(words_list)
                                # cv2.putText(clone,'Sign'+str(predict_sign), (100, 300), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))

                            '''print(final_sign, " ", prev_sign," ",prob)
                            #if prev_sign and prob>=90:
                            if final_sign == prev_sign:
                                count+=1
                            else:
                                count=0

                            if count>15:
                                Thread(target=say_sign, args=(final_sign,)).start()
                                count=0

                            prev_sign=final_sign
                            print(count)'''



                        else:
                            if words_list is not None:
                                # Thread(target=say_sign,args=(words_list,)).start()
                                words_list.clear()

            elif method==2:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lh = cv2.getTrackbarPos("LH", "Tracking")
                ls = cv2.getTrackbarPos("LS", "Tracking")
                lv = cv2.getTrackbarPos("LV", "Tracking")
                uh = cv2.getTrackbarPos("UH", "Tracking")
                us = cv2.getTrackbarPos("US", "Tracking")
                uv = cv2.getTrackbarPos("UV", "Tracking")

                cv2.putText(clone, "Put your hand in the rectangle", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                cv2.putText(clone, "Adjust the values using trackbar", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0))
                cv2.putText(clone, "Press the key of the sample", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                cv2.putText(clone, "Press esc to exit.", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

                l_b = np.array([lh, ls, lv])
                u_b = np.array([uh, us, uv])

                mask = cv2.inRange(hsv, l_b, u_b)
                mask = cv2.bitwise_not(mask)
                res = cv2.bitwise_and(roi, roi, mask=mask)
                res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

                (_, cnts, _) = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts)>0:
                    max_cont = max(cnts, key=cv2.contourArea)
                    if max_cont is not None:
                        mask = cv2.drawContours(res, [max_cont + (r, t)], -1, (0, 0, 255))
                        cv2.imshow("Threshold", mask)
                        mask = np.zeros(res.shape, dtype="uint8")
                        cv2.drawContours(mask, [max_cont], -1, 255, -1)
                        res = cv2.bitwise_and(res, res, mask=mask)
                        cv2.imshow('mask', mask)

                        high_thresh, thresh_im = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        lowThresh = 0.5 * high_thresh
                        res = cv2.Canny(res, lowThresh, high_thresh)

                        # CNN Model
                        if res is not None and cv2.contourArea(max_cont) > 1000:
                            final_res = cv2.resize(res, (100, 100))
                            final_res = np.array(final_res)
                            final_res = final_res.reshape((-1, 100, 100, 1))
                            final_res.astype('float32')
                            final_res = final_res / 255.0
                            output = cnn.predict(final_res)
                            prob = np.amax(output)
                            sign = np.argmax(output)
                            final_sign = visual_dict[sign]
                            cv2.putText(clone, 'Sign ' + str(final_sign), (10, 200), cv2.FONT_HERSHEY_COMPLEX, 2,
                                        (0, 0, 255))

                            # print(count)
                            count += 1
                            if (count > 10 and count <= 50):
                                if (prob * 100 > 95):
                                    result_list.append(final_sign)
                                    # print(sign, prob)
                            elif (count > 50):
                                count = 0
                                if len(result_list):
                                    predict_sign = (max(set(result_list), key=result_list.count))
                                    result_list = []
                                    if prev_sign is not None:
                                        if prev_sign != predict_sign:
                                            # print(words_list)
                                            words_list += str(predict_sign)
                                            Thread(target=say_sign, args=(predict_sign,)).start()
                                    else:
                                        Thread(target=say_sign, args=(predict_sign,)).start()
                                        # prev_sign=predict_sign
                                    prev_sign = predict_sign
                                # print(words_list)
                                # cv2.putText(clone,'Sign'+str(predict_sign), (100, 300), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))

                        else:
                            if words_list is not None:
                                # Thread(target=say_sign,args=(words_list,)).start()
                                words_list.clear()

            cv2.rectangle(clone, (l, t), (r, b), (0, 255, 0), 2)
            num_frames += 1
            cv2.imshow("Video Feed", clone)

        else:
            messagebox.showerror("error","Can't grab frame")
            break

        k=cv2.waitKey(1)& 0xFF
        if (k==27):
           break


    cam.release()
    cv2.destroyAllWindows()

#pred_main()