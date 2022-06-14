# encoding:utf-8
import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
mp_objectron = mp.solutions.objectron            # mediapipe 物體偵測

cap = cv2.VideoCapture(0)

def text(text):      # 建立顯示文字的函式
    global img       # 設定 img 為全域變數
    org = (0,50)     # 文字位置
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1                        # 文字尺寸
    color = (255,255,255)                # 顏色
    thickness = 5                        # 文字外框線條粗細
    lineType = cv2.LINE_AA               # 外框線條樣式
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType) # 放入文字

def find_angle(lmslist, img, p1, p2, p3, draw=True):
    x1, y1 = lmslist[p1][1], lmslist[p1][2]
    x2, y2 = lmslist[p2][1], lmslist[p2][2]
    x3, y3 = lmslist[p3][1], lmslist[p3][2]
    if x1==0 or x2==0 or x3==0 or y1==0 or y2==0 or y3==0:
        return -1

    # 使用三角函数公式获取3个点p1-p2-p3，以p2为角的角度值，0-180度之间
    angle = int(math.degrees(math.atan2(y1 - y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2)))
    if angle < 0:
        angle = angle + 360
    if angle > 180:
        angle = 360 - angle

    if draw:
        cv2.circle(img, (x1, y1), 20, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 30, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 20, (0, 255, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255, 3))
        cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255, 3))
        cv2.putText(img, str(angle), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)

    return angle

def find_positions(pose_landmarks, img):
    lmslist = []
    if pose_landmarks:
        for id, lm in enumerate(pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmslist.append([id, cx, cy])

    return lmslist

fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA               # 印出文字的邊框

# 啟用姿勢偵測
with (mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose,
    mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.99,
    model_name='Shoe') as objectron):

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        w, h = 540, 350                               # 影像尺寸
        img = cv2.resize(img,(w,h))                   # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        
        pose_results = pose.process(img2)             # 取得姿勢偵測結果
        obj_results = objectron.process(img2)         # 取得物體偵測結果
        
        det_shoe = False;
        
        det_waist_pose = "None"
        det_elbow_pose = "None"
            
        if pose_results.pose_landmarks != None:
            mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            pose_points = find_positions(pose_results.pose_landmarks, img2)
            
            waist_angle = find_angle(pose_points, img, 11, 23, 25)
            if waist_angle>=0:
                if waist_angle <= 55:
                    det_waist_pose = "Sitting"
                elif waist_angle >= 120:
                    det_waist_pose = "Lie Down"
                else:
                    det_waist_pose = "Half Sitting"
            
            elbow_angle = find_angle(pose_points, img, 11, 13, 15)
            if elbow_angle>=0:
                if elbow_angle <= 90:
                    det_elbow_pose = "Look At Book/Phone"
                
        # 標記所偵測到的物體
        if obj_results.detected_objects:
            det_shoe = True
            for detected_object in obj_results.detected_objects:
                mp_drawing.draw_landmarks(img, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(img, detected_object.rotation, detected_object.translation)
        
        if det_shoe==True and det_waist_pose=="Lie Down":
            text("So Dirty! Please Drop Shoes!")
        elif det_elbow_pose=="Look At Book/Phone" and det_waist_pose=="Lie Down":
            text("Put Down Your Book or Phone!")
        else:
            text(det_waist_pose)
        
        cv2.imshow('W-Link', img)
        
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()