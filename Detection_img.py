from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import os

# set local path of model
face_model = r'face_detector'
mask_model = r'model.h5'

# minimum probability to filter weak detections
CONFIDENCE = 0.5

        

print("loading face detector model")
prototxtPath = os.path.join(face_model,"deploy.prototxt")
weightsPath = os.path.join(face_model,"res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("loading face mask detector model...")
maskNet = load_model("model.h5")

#%%
path = r"image\0.jpg"
# path = r"image\2.jpg"
# path = r"image\4.jpg"
img = cv2.imread(path)

(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img,
                             scalefactor=1.0,  # 各通道數值的縮放比例
                             size=(300, 300),  # 輸出圖像的尺寸 (W,H)
                             mean=(104.0, 177.0, 123.0),  # 各通道減去的值，以降低光照的影響
                             swapRB=True,  # 減均值順序是 (R,G,B)
)

faceNet.setInput(blob)
detections = faceNet.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0,0,i,2]
		
    if confidence > CONFIDENCE:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        
        face = img[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
        pred = maskNet.predict(face)
        mask, withoutMask = pred[0]
        
        label = "Mask %4.2f"%mask if mask > withoutMask else "No Mask %4.2f"%withoutMask
        color = (0, 255, 0) if label.find("Mask")==0 else (0, 0, 255)
        cv2.rectangle(img, (startX+20, startY), (endX, endY), color, 2)
        cv2.putText(img, label, (startX+10,endY+25), cv2.FONT_HERSHEY_DUPLEX , 1, color)
    
cv2.imshow('Frame', img)
cv2.imwrite(r"image\test.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


