import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
import os
import numpy as np
import time


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
        
        self.captured_photo = None
        self.photo_shape_pts = None
        self.was_3fingers_down = False
        self.is_grabbing = False
        self.grab_offset = [0, 0]
        self.photo_pos = [0, 0]
        self.photo_size = (0, 0)
        self.photo_mask = None
        self.capture_timer = 0
        self.capture_shape_pts = None
        self.smooth_pos = [0, 0]
        self.trash_zone = None
        self.trash_timer = 0
        
        self.cap = None
        self.running = False
        self.hand_landmarker = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.canvas_w = 0
        self.canvas_h = 0
        
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
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO
        )
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
        
        self.hand_info = tk.Label(footer, text="MAOS: 0 | FPS: 0", 
                                  font=("Consolas", 10), bg="#0d0d0d", fg="#003300")
        self.hand_info.pack(side=tk.LEFT, padx=10)
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            self.status_label.config(text="[ERRO]", fg="#ff0000")
            return
        
        self.running = True
        self.status_label.config(text="[CONECTADO]", fg="#00ff41")
        self.process_frame()
        
    def is_finger_down(self, lm, tip_id, pip_id):
        return lm[tip_id].y > lm[pip_id].y
        
    def check_3fingers_down(self, lm):
        return self.is_finger_down(lm, 12, 10) and self.is_finger_down(lm, 16, 14) and self.is_finger_down(lm, 20, 18)
        
    def is_pinching(self, lm, w, h):
        thumb = (int(lm[4].x * w), int(lm[4].y * h))
        index = (int(lm[8].x * w), int(lm[8].y * h))
        dist = ((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2) ** 0.5
        mx = thumb[0] * 0.5 + index[0] * 0.5
        my = thumb[1] * 0.5 + index[1] * 0.5
        return dist < 50, (int(mx), int(my))
        
    def get_right_hand(self, result):
        if not result.hand_landmarks or len(result.hand_landmarks) < 2:
            return None, None
        
        right_hand = None
        other_hand = None
        
        for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            if handedness[0].category_name == "Right":
                right_hand = landmarks
            else:
                other_hand = landmarks
        
        if right_hand is None:
            right_hand = result.hand_landmarks[0]
            other_hand = result.hand_landmarks[1]
        
        return right_hand, other_hand
        
    def process_frame(self):
        if not self.running:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.process_frame)
            return
            
        frame = cv2.flip(frame, 1)
        
        now = time.time()
        self.frame_count += 1
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            self.canvas_w = canvas_w
            self.canvas_h = canvas_h
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(time.time() * 1000)
            result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            img = cv2.resize(frame, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
            h, w, _ = img.shape
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output = cv2.merge([gray // 3, gray, gray // 3])
            
            three_fingers_down = False
            shape_pts = None
            pinch_pos = None
            is_pinching_now = False
            
            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                self.hand_info.config(text=f"MAOS: {num_hands} | FPS: {self.fps}")
                
                for hand_landmarks in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                    
                    pinching, pinch_center = self.is_pinching(hand_landmarks, w, h)
                    if pinching:
                        is_pinching_now = True
                        pinch_pos = pinch_center
                    
                    for start, end in self.HAND_CONNECTIONS:
                        cv2.line(output, pts[start], pts[end], (0, 255, 65), 1, cv2.LINE_AA)
                    
                    for pt in pts:
                        cv2.circle(output, pt, 2, (0, 255, 65), -1)
                
                if num_hands >= 2:
                    right_hand, other_hand = self.get_right_hand(result)
                    
                    if right_hand and other_hand:
                        right_pts = [(int(lm.x * w), int(lm.y * h)) for lm in right_hand]
                        other_pts = [(int(lm.x * w), int(lm.y * h)) for lm in other_hand]
                        
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
            else:
                self.hand_info.config(text=f"MAOS: 0 | FPS: {self.fps}")
            
            if three_fingers_down and not self.was_3fingers_down and shape_pts is not None:
                self.capture_timer = time.time()
                self.capture_shape_pts = shape_pts.copy()
            
            if not three_fingers_down and self.capture_timer > 0:
                self.capture_timer = 0
                self.capture_shape_pts = None
            
            if self.capture_timer > 0 and self.capture_shape_pts is not None:
                elapsed = time.time() - self.capture_timer
                if elapsed >= 0.5:
                    self.capture_photo_in_shape(img, self.capture_shape_pts)
                    self.capture_timer = 0
                    self.capture_shape_pts = None
                elif self.capture_shape_pts is not None:
                    remaining = 0.5 - elapsed
                    cv2.putText(output, f"{remaining:.1f}", (w // 2 - 15, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 65), 1, cv2.LINE_AA)
            
            self.was_3fingers_down = three_fingers_down
            
            if self.captured_photo is not None and is_pinching_now and pinch_pos:
                if not self.is_grabbing:
                    self.grab_offset[0] = pinch_pos[0] - self.photo_pos[0]
                    self.grab_offset[1] = pinch_pos[1] - self.photo_pos[1]
                    self.smooth_pos[0] = pinch_pos[0]
                    self.smooth_pos[1] = pinch_pos[1]
                self.is_grabbing = True
                target_x = pinch_pos[0] - self.grab_offset[0]
                target_y = pinch_pos[1] - self.grab_offset[1]
                self.smooth_pos[0] = self.smooth_pos[0] * 0.3 + target_x * 0.7
                self.smooth_pos[1] = self.smooth_pos[1] * 0.3 + target_y * 0.7
                self.photo_pos[0] = int(self.smooth_pos[0])
                self.photo_pos[1] = int(self.smooth_pos[1])
            elif not is_pinching_now and self.is_grabbing:
                self.trash_timer = 0
                self.is_grabbing = False
            
            if self.captured_photo is not None:
                self.draw_trash_bin(output)
                self.draw_captured_photo(output)
                    
            img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk
            
        self.root.after(1, self.process_frame)
        
    def capture_photo_in_shape(self, img, shape_pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shape_pts], 255)
        
        photo = img.copy()
        photo[mask == 0] = [10, 10, 10]
        
        pts_array = shape_pts.reshape(-1, 2)
        x_min, y_min = pts_array.min(axis=0)
        x_max, y_max = pts_array.max(axis=0)
        
        crop = photo[y_min:y_max, x_min:x_max].copy()
        
        self.captured_photo = crop
        self.photo_size = (x_max - x_min, y_max - y_min)
        self.photo_pos = [x_min, y_min]
        self.photo_shape_pts = shape_pts - np.array([x_min, y_min])
        
    def draw_captured_photo(self, img):
        if self.captured_photo is None:
            return
        
        x = int(self.photo_pos[0])
        y = int(self.photo_pos[1])
        pw, ph = self.photo_size
        
        h_img, w_img, _ = img.shape
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + pw)
        y2 = min(h_img, y + ph)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        region = self.captured_photo[src_y1:src_y2, src_x1:src_x2]
        
        if region.shape[0] > 0 and region.shape[1] > 0:
            mask = np.zeros(region.shape[:2], dtype=np.uint8)
            shifted_pts = self.photo_shape_pts - np.array([src_x1, src_y1])
            cv2.fillPoly(mask, [shifted_pts.astype(np.int32)], 255)
            
            idx = mask > 0
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            tinted = np.stack([gray_region // 3, gray_region, gray_region // 3], axis=-1).astype(np.uint8)
            
            img[y1:y2, x1:x2][idx] = tinted[idx]
        
    def draw_trash_bin(self, img):
        h, w, _ = img.shape
        bx = 30
        by = 30
        bw = 70
        bh = 90
        
        self.trash_zone = (bx, by, bx + bw, by + bh)
        
        in_trash = False
        if self.is_grabbing:
            px = self.photo_pos[0] + self.photo_size[0] // 2
            py = self.photo_pos[1] + self.photo_size[1] // 2
            if bx + 10 <= px <= bx + bw - 10 and by + 10 <= py <= by + bh - 10:
                in_trash = True
        
        if in_trash:
            self.trash_timer += 1
            if self.trash_timer > 30:
                self.captured_photo = None
                self.photo_shape_pts = None
                self.photo_size = (0, 0)
                self.photo_pos = [0, 0]
                self.trash_timer = 0
                return
        else:
            self.trash_timer = 0
        
        progress = self.trash_timer / 30.0
        
        if in_trash:
            color = (0, int(100 + 155 * progress), 255)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)
            fill_h = int(bh * progress)
            overlay = img[by + bh - fill_h:by + bh, bx:bx + bw].copy()
            cv2.rectangle(img, (bx, by + bh - fill_h), (bx + bw, by + bh), (0, 80, 200), -1)
        else:
            color = (0, 60, 0)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)
        
        cv2.line(img, (bx + 5, by - 3), (bx + bw - 5, by - 3), color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (bx + 25, by - 10), (bx + 45, by - 3), color, 2, cv2.LINE_AA)
        
        for i in range(3):
            lx = bx + 18 + i * 12
            cv2.line(img, (lx, by + 12), (lx, by + bh - 12), color, 2, cv2.LINE_AA)
        
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
