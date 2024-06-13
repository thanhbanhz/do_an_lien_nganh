import cv2
import numpy as np
from main_src.utils import *
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

model = YOLO(r"D:\do_an_lien_nganh\do_an_lien_nganh\weight\best.pt")
tracker = DeepSort(max_age=30)
tich_luy = 0
state_all = {}
ok = {}

colors = np.random.randint(0,255, size=(5,3 ))
y_line = 450

cap = cv2.VideoCapture(r"video_test/light2.mp4")
ret, frame = cap.read()
x_all,y_all,w_all,h_all = (170, 275, 800, 414)
roi = (600, 0, 95, 46)
x, y, w, h = roi
i = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))
process = tqdm(total=total_frames)
while True:
    if i %3 == 0 :
        i+=1
    ret, big_frame = cap.read()
    all_frame = big_frame.copy()
    cv2.rectangle(all_frame, (x_all, y_all), (x_all + w_all, y_all + h_all), (255,255,0), 3)
    if not ret:
        break
    
    frame = big_frame[y:y+h, x:x+w, :]
    red, tich_luy,_ = is_red(frame, tich_luy_hien_tai=tich_luy)
    text = 'red' if red else 'green'
    light_color = (0,0,255) if red else (0,255,0)

    big_frame = cv2.cvtColor(big_frame[y_all:y_all+h_all,x_all:x_all+w_all],cv2.COLOR_BGR2RGB)
    result = model.predict(big_frame,conf = 0.35,verbose = False)
    
    if len(result):
        result = result[0]
        names = result.names
        detect = []
        for box in result.boxes:
            x1,y1,x2,y2 = list(map(int,box.xyxy[0]))
            conf = box.conf.item()
            cls = int(box.cls.item())
            detect.append([[x1,y1,x2-x1,y2-y1],conf,cls])
           
        tracks = tracker.update_tracks(detect, frame = big_frame)
    
        # Vẽ lên màn hình các khung chữ nhật kèm ID
        for i,track in enumerate(tracks):
            if track.is_confirmed() and track.det_conf :
                
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                x1 += x_all
                x2 += x_all
                y1 += y_all
                y2 += y_all
                yc = y1+abs(y2-y1)//2
                
                track_id = track.track_id
                name = names[track.det_class]
                color = (colors[track.det_class]).tolist()
                is_ok = ok.get(track_id,None)
                
                label = None
                if is_ok:
                    label = "k vuot" if ok[track_id] == 2 else 'vuot'
                else:
                    state = state_all.get(track_id)
                    
                    if state is None:
                        if yc < y_line:
                            ok[track_id] = 2
                            label = "k vuot"
                        else:
                            state_all[track_id] = red
                    else:
                        if yc < y_line:
                            ok[track_id] = 1 if state_all[track_id] else 2
                            label = "k vuot" if ok[track_id] == 2 else 'vuot'
                        else:
                            state_all[track_id] = red

               
                if label is None:
                    label = f"{track_id} : {name[:3]} {round(track.det_conf, 2)}"
                else:
                    color = (0,0,255) if label == 'vuot' else (0,255,0)
                    
                
                cv2.rectangle(all_frame, (x1, y1), (x2, y2),  color, 2)
                cv2.rectangle(all_frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1),  color, -1)
                cv2.putText(all_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.line(all_frame,(0,y_line),(all_frame.shape[1],y_line),(0,0,255),2)
    all_frame = draw_text(all_frame,text,color=light_color)
    cv2.imshow("frame", all_frame)
    out.write(all_frame)
    process.update(1)

    if cv2.waitKey(8) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
