import cv2
import numpy as np

img = cv2.imread('images/test_sample1.jpg', 0) #read the image
width = 389
height = 550
img = cv2.resize(img,(width,height)) #resize the image based on predefined width and height
cv2.imshow('before_rot_1', img) #show the resized image
img = cv2.bitwise_not(img)  #getting the negative of the image
edges = cv2.Canny(img,50,150,apertureSize = 3) #use canny edge detector to detect edges to be used to detect hough lines
lines = cv2.HoughLines(edges,1,np.pi/180,100) #detect the horizontal lines in the image
cv2.imshow('before_rot_2', img)
rho = lines[0][0][0]
theta = lines[0][0][1]
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b)) #get the x component of the first point on the line
y1 = int(y0 + 1000*(a))  #get the y component of the first point on the line
x2 = int(x0 - 1000*(-b)) #get the x component of the second point on the line
y2 = int(y0 - 1000*(a))  #get the y component of the second point on the line
angle = np.arctan((y2-y1)/(x2-x1))*180/(22/7) #get the rotation angle by getting the slope of the line from the two points and getting the arctan of the slope to get the angle
image_center = tuple(np.array(img.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR) #rotating the image with the calculated angle
cv2.imshow('rotated',img)                                #show the rotated image
img = cv2.threshold(img,230,255,cv2.THRESH_BINARY)[1]    #threshold to show the circles only and hide all the unwanted features
circles = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)  #detect the circles in the image using connected components
circles = circles[2]  #get the x,y coordinates of all the detected circles in order based on the y position of each circle
circles = circles[1:23] #neglecting the first connected component as it represents the background
cv2.imshow('detected circles',img)  #show the detected circles in the image
answers = {}  #create array to store the answers based on the x,y position of the circles

# detect the the gender, since the first circle detected is for gender
if 0.73*width <= circles[0][0] <= 0.8*width:
    answers[0] = "male"
elif 0.82*width <= circles[0][0] <= 0.86*width:
    answers[0] = "female"

#detect the semester, since the second circle detected is for semester
if 0.31*width <= circles[1][0] <= 0.35*width:
    answers[1] = "fall"
elif 0.47*width <= circles[1][0] <= 0.52*width:
    answers[1] = "spring"
elif 0.62*width <= circles[1][0] <= 0.68*width:
    answers[1] = "summer"

#detect the program, since the third circle detected is for program, in this case we have to compare the x,y position of the circle to detect the program
if 0.26*width <= circles[2][0] <= 0.29*width:
    if 0.184*height <= circles[2][1] <= 0.199*height:
        answers[2] = "MCTA"
    elif 0.2*height <= circles[2][1] <= 0.215*height:
        answers[2] = "LAAR"
elif 0.33*width <= circles[2][0] <= 0.4*width:
    if 0.184*height <= circles[2][1] <= 0.199*height:
        answers[2] = "ENVR"
    elif 0.2*height <= circles[2][1] <= 0.215*height:
        answers[2] = "MATL"
elif 0.4*width <= circles[2][0] <= 0.46*width:
    if 0.184*height <= circles[2][1] <= 0.199*height:
        answers[2] = "BLDG"
    elif 0.2*height <= circles[2][1] <= 0.215*height:
        answers[2] = "CISE"
elif 0.49*width <= circles[2][0] <= 0.54*width:
    if 0.184*height <= circles[2][1] <= 0.199*height:
        answers[2] = "CESSs"
    elif 0.2*height <= circles[2][1] <= 0.215*height:
        answers[2] = "HAUD"
elif 0.57*width <= circles[2][0] <= 0.62*width:  #we no longer need to detect the y coordinate because all the remaining circles are on the same line
    answers[2] = "ERGY"
elif 0.64*width <= circles[2][0] <= 0.7*width:
    answers[2] = "COMM"
elif 0.73*width <= circles[2][0] <= 0.79*width:
    answers[2] = "MANF"

#detect the answers of the remaining questions since all the remaining have the same x position ranges
for i in range(3,22):
    if 0.655*width <= circles[i][0] <= 0.717*width:
        answers[i] = "strongly agree"
    elif 0.724*width <= circles[i][0] <= 0.769*width:
        answers[i] = "agree"
    elif 0.78*width <= circles[i][0] <= 0.821*width:
        answers[i] = "neutral"
    elif 0.842*width <= circles[i][0] <= 0.884*width:
        answers[i] = "disagree"
    elif 0.898*width <= circles[i][0] <= 0.946*width:
        answers[i] = "strongly disagree"

#defining the questions to be answered
questions = ["Gender: ", "Semester: ", "Program: ", "The teaching on this course/module is intellectually stimulating: ",
             "Matters are explained well in the teaching sessions: ", "the teaching methods used helped to learn: ",
             "Lectures were good at explaining things: ", "The course/module was academically challenging: ",
             "I am aware of the course/module learning outcomes: ", "The assessment requirements were clear: ",
             "I feel well supported on this course/module: ","Feedback on summative work was provided within the time specified: ",
             "The work load for this course/module is manageable: ", "The assessments completed so far stimulated my learning: ",
             "The course/module was well organized and ran smoothly:", "The course/module focused on what was set out in the student guide: ",
             "I have been able to contact staff when I needed to: ", "The course/module materials on Moodle are helpful in supporting my learning: ",
             "The library resources for the course/module including its digital resources met my needs: ",
             "I am satisfied with the quality of classroom facilities for this course/module: ",
             "Overall I was satisfied with my experience of this course/module: ",
             "I would recommend this course/module to another student: "]

output_file = open(r"output.txt","w")  #opening the text file to begin writing in it
for i in range(22):
    l = questions[i] + answers[i]+'\n' #concatenating the questions with their answers
    output_file.write(l)               #writing the concatenated line to the text file
output_file.close()                    #closing the file after writing all the questions and their corresponding answers
cv2.waitKey(0)                         #wait till a key is pressed
cv2.destroyAllWindows()                #close all opened windows and terminate the program