import os, sys, cv2, requests, glob
#from google.colab.patches import cv2_imshow
from PIL import Image 
import PIL 
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
os.system("git clone https://github.com/bxck75/Face_Zoo.git")

haar_path = os.path.join(ROOT_DIR,'Face_Zoo')

face_detector = cv2.CascadeClassifier(os.path.join(haar_path,'haarcascade_frontalface_default.xml'))
eye_detector = cv2.CascadeClassifier(os.path.join(haar_path,'haarcascade_eye.xml'))

"""##Get a groupfoto from internet
and read it
"""
group_img_url =''
if len(sys.argv) > 1:
    group_img_url = sys.argv[1]
else:
    group_img_url =  os.path.join(haar_path,'nintchdbpict000299847684-1.jpg')

name_split = group_img_url.split('/')
img_file_name = name_split.pop()
img_name = img_file_name.replace('.jpg','').replace('.png','').replace('.webp','')
"""
response = requests.get(group_img_url)
with open(img_file_name, 'wb') as f:
    f.write(response.content)
"""
img_path = img_file_name
faces_out = os.path.join(ROOT_DIR, img_name)
os.makedirs(faces_out, exist_ok = True)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
faces_result = face_detector.detectMultiScale(gray, 1.3, 5) 
i=0
for (x,y,w,h) in faces_result: 
  img = cv2.rectangle(img,(x,y),(x+w,y+h),(300,0,0),2) 
  roi_gray = gray[y:y+h, x:x+w] 
  roi_color = img[y:y+h, x:x+w] 
  eyes = eye_detector.detectMultiScale(roi_gray) 
  if len(eyes) > 1:
    #cv2.imshow('col_img', roi_color)
    pilim = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(pilim)
    # save a image using extension
    im_pil.save(os.path.join(faces_out, "face_" + str(i) + ".jpg"))
    i += 1
#  for (ex,ey,ew,eh) in eyes: 
#    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#cv2.imshow('img',img) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

import glob
print(str(len(glob.glob(faces_out+'/*.jpg'))) + " faces found")