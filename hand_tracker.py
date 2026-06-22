import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
import os
import numpy as np


class HandTracker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Tracker")
        self.root.geometry("900x700")
        self.root.configure(bg="#0a0a0a")
        
        self.HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        
        self.FINGER_TIPS = [4, 8]
        
        self.cap = None
        self.running = False
        self.hand_landmarker = None
        
        self.setup_ui()
        self.download_model()
        self.start_camera()
        
    def download_model(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
    def setup_ui(self):
        header = tk.Frame(self.root, bg="#0d0d0d", height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="HAND_TRACKER v1.0", font=("Consolas", 12, "bold"), 
                 bg="#0d0d0d", fg="#00ff41").pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(header, text="[CONECTADO]", 
                                     font=("Consolas", 10), bg="#0d0d0d", fg="#00ff41")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        self.canvas = tk.Canvas(self.root, bg="#0a0a0a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        footer = tk.Frame(self.root, bg="#0d0d0d", height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        self.hand_info = tk.Label(footer, text="MAOS: 0", 
                                  font=("Consolas", 10), bg="#0d0d0d", fg="#003300")
        self.hand_info.pack(side=tk.LEFT, padx=10)
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="[ERRO]", fg="#ff0000")
            return
        
        self.running = True
        self.status_label.config(text="[CONECTADO]", fg="#00ff41")
        self.process_frame()
        
    def process_frame(self):
        if not self.running:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(100, self.process_frame)
            return
            
        frame = cv2.flip(frame, 1)
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = self.hand_landmarker.detect(mp_image)
            
            img = cv2.resize(frame, (canvas_w, canvas_h))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output = cv2.merge([gray//4, gray//2, gray//4])
            
            hand_tips = []
            
            if result.hand_landmarks:
                self.hand_info.config(text=f"MAOS: {len(result.hand_landmarks)}")
                
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    h, w, _ = output.shape
                    
                    pts = []
                    for lm in hand_landmarks:
                        pts.append((int(lm.x * w), int(lm.y * h)))
                    
                    tips = [pts[i] for i in self.FINGER_TIPS]
                    hand_tips.append(tips)
                    
                    for start, end in self.HAND_CONNECTIONS:
                        cv2.line(output, pts[start], pts[end], (0, 255, 65), 1, cv2.LINE_AA)
                    
                    for pt in pts:
                        cv2.circle(output, pt, 2, (0, 255, 65), -1)
                
                if len(hand_tips) == 2:
                    self.draw_shape_between_tips(output, hand_tips)
            else:
                self.hand_info.config(text="MAOS: 0")
                    
            img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk
            
        self.root.after(30, self.process_frame)
        
    def draw_shape_between_tips(self, img, hand_tips):
        hand1_tips = hand_tips[0]
        hand2_tips = hand_tips[1]
        
        thumb1 = hand1_tips[0]
        index1 = hand1_tips[1]
        thumb2 = hand2_tips[0]
        index2 = hand2_tips[1]
        
        if thumb1[0] > thumb2[0]:
            thumb1, thumb2 = thumb2, thumb1
            index1, index2 = index2, index1
        
        pts = [thumb1, index1, index2, thumb2]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        cv2.polylines(img, [pts], True, (0, 255, 65), 2, cv2.LINE_AA)
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app = HandTracker()
    app.run()
