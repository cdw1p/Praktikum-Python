import urllib.request
import requests
import cv2
import numpy as np
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def store_raw_images():
    positive_images_link = ''
    positive_image_urls = urllib.request.urlopen(positive_images_link).read().decode()
    pic_num = 1

    if not os.path.exists('positive'):
        os.makedirs('positive')
        
    for i in positive_image_urls.split('\n'):
        try:
            print("(" + str(pic_num) + ") " + i)
            response = requests.get(i, timeout=5)

            if response.status_code == 200:
                with open("positive/"+str(pic_num)+".jpg", 'wb') as f:
                    f.write(response.content)

            img = cv2.imread("positive/"+str(pic_num)+".jpg",1)
            cv2.imwrite("positive/"+str(pic_num)+".jpg",img)
            pic_num += 1

        except Exception as e:
            print(str(e))  

store_raw_images()