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
        
        self.captured_photo = None
        self.photo_shape_pts = None
        self.was_3fingers_down = False
        self.frame_for_capture = None
        
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
        
        self.hand_info = tk.Label(footer, text="MAOS: 0 | FOTO: NAO", 
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
        
    def is_finger_down(self, landmarks, tip_id, pip_id):
        return landmarks[tip_id].y > landmarks[pip_id].y
        
    def check_3fingers_down(self, landmarks):
        middle = self.is_finger_down(landmarks, 12, 10)
        ring = self.is_finger_down(landmarks, 16, 14)
        pinky = self.is_finger_down(landmarks, 20, 18)
        return middle and ring and pinky
        
    def get_right_hand(self, result):
        if not result.hand_landmarks or len(result.hand_landmarks) < 2:
            return None, None, None
        
        hands_data = []
        for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handednesses)):
            label = handedness[0].category_name
            hands_data.append((i, label, landmarks))
        
        right_hand = None
        other_hand = None
        for idx, label, landmarks in hands_data:
            if label == "Right":
                right_hand = landmarks
            else:
                other_hand = landmarks
        
        if right_hand is None and len(hands_data) >= 2:
            right_hand = hands_data[0][2]
            other_hand = hands_data[1][2]
        
        return right_hand, other_hand, len(hands_data)
        
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
            three_fingers_down = False
            shape_pts = None
            
            if result.hand_landmarks and len(result.hand_landmarks) >= 2:
                right_hand, other_hand, count = self.get_right_hand(result)
                
                if right_hand and other_hand:
                    self.hand_info.config(text=f"MAOS: {count} | FOTO: {'SIM' if self.captured_photo is not None else 'NAO'}")
                    
                    h, w, _ = output.shape
                    
                    right_pts = []
                    for lm in right_hand:
                        right_pts.append((int(lm.x * w), int(lm.y * h)))
                    
                    other_pts = []
                    for lm in other_hand:
                        other_pts.append((int(lm.x * w), int(lm.y * h)))
                    
                    if self.check_3fingers_down(right_hand):
                        three_fingers_down = True
                    
                    thumb_r = right_pts[4]
                    index_r = right_pts[8]
                    thumb_o = other_pts[4]
                    index_o = other_pts[8]
                    
                    if thumb_r[0] > thumb_o[0]:
                        thumb_r, thumb_o = thumb_o, thumb_r
                        index_r, index_o = index_o, index_r
                    
                    shape_pts = np.array([thumb_r, index_r, index_o, thumb_o], np.int32)
                    shape_pts = shape_pts.reshape((-1, 1, 2))
                    
                    cv2.polylines(output, [shape_pts], True, (0, 255, 65), 2, cv2.LINE_AA)
                    
                    for lm in right_hand:
                        pt = (int(lm.x * w), int(lm.y * h))
                        cv2.circle(output, pt, 2, (0, 255, 65), -1)
                    for lm in other_hand:
                        pt = (int(lm.x * w), int(lm.y * h))
                        cv2.circle(output, pt, 2, (0, 255, 65), -1)
            else:
                self.hand_info.config(text="MAOS: 0 | FOTO: NAO")
            
            if three_fingers_down and not self.was_3fingers_down and shape_pts is not None:
                self.capture_photo_in_shape(img, shape_pts)
            
            self.was_3fingers_down = three_fingers_down
            
            if self.captured_photo is not None:
                self.draw_captured_photo(output)
                    
            img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk
            
        self.root.after(30, self.process_frame)
        
    def capture_photo_in_shape(self, img, shape_pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shape_pts], 255)
        
        photo = img.copy()
        photo[mask == 0] = [10, 10, 10]
        
        self.captured_photo = photo
        self.photo_shape_pts = shape_pts.copy()
        
    def draw_captured_photo(self, img):
        if self.captured_photo is None or self.photo_shape_pts is None:
            return
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.photo_shape_pts], 255)
        
        photo_resized = cv2.resize(self.captured_photo, (img.shape[1], img.shape[0]))
        
        idx = mask > 0
        img[idx] = photo_resized[idx]
        
        cv2.polylines(img, [self.photo_shape_pts], True, (0, 255, 65), 2, cv2.LINE_AA)
        
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
