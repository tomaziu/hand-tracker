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
        
        self.photos = []
        self.was_3fingers_down = False
        self.is_grabbing = False
        self.grab_offset = [0, 0]
        self.grabbed_photo_idx = -1
        self.capture_timer = 0
        self.capture_shape_pts = None
        self.smooth_pos = [0, 0]
        
        self.flash_effect = 0
        self.flash_particles = []
        
        self.is_zooming = False
        self.zoom_photo_idx = -1
        self.zoom_initial_dist = 0
        self.zoom_initial_scale = 1.0
        
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
        
        self.hand_info = tk.Label(footer, text="MAOS: 0 | FPS: 0 | FOTOS: 0", 
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
        
    def find_photo_at_pos(self, px, py):
        for i in range(len(self.photos) - 1, -1, -1):
            photo = self.photos[i]
            x, y = photo['pos']
            pw, ph = photo['size']
            if x <= px <= x + pw and y <= py <= y + ph:
                return i
        return -1
        
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
            two_pinches = False
            pinch_positions = []
            
            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                self.hand_info.config(text=f"MAOS: {num_hands} | FPS: {self.fps} | FOTOS: {len(self.photos)}")
                
                for hand_landmarks in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                    
                    pinching, pinch_center = self.is_pinching(hand_landmarks, w, h)
                    if pinching:
                        is_pinching_now = True
                        pinch_pos = pinch_center
                        pinch_positions.append(pinch_center)
                    
                    for start, end in self.HAND_CONNECTIONS:
                        cv2.line(output, pts[start], pts[end], (0, 255, 65), 1, cv2.LINE_AA)
                    
                    for pt in pts:
                        cv2.circle(output, pt, 2, (0, 255, 65), -1)
                
                if len(pinch_positions) == 2:
                    two_pinches = True
                
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
                self.hand_info.config(text=f"MAOS: 0 | FPS: {self.fps} | FOTOS: {len(self.photos)}")
            
            if three_fingers_down and not self.was_3fingers_down and shape_pts is not None:
                self.capture_timer = time.time()
                self.capture_shape_pts = shape_pts.copy()
            
            if not three_fingers_down and self.capture_timer > 0:
                self.capture_timer = 0
                self.capture_shape_pts = None
            
            if self.capture_timer > 0 and self.capture_shape_pts is not None:
                elapsed = time.time() - self.capture_timer
                if elapsed >= 0.5:
                    self.add_photo(img, self.capture_shape_pts)
                    self.capture_timer = 0
                    self.capture_shape_pts = None
                    self.flash_effect = 1.0
                    for _ in range(20):
                        self.flash_particles.append({
                            'x': np.random.randint(0, w),
                            'y': np.random.randint(0, h),
                            'vx': np.random.uniform(-3, 3),
                            'vy': np.random.uniform(-3, 3),
                            'life': 1.0,
                            'size': np.random.randint(2, 6)
                        })
                elif self.capture_shape_pts is not None:
                    remaining = 0.5 - elapsed
                    cv2.putText(output, f"{remaining:.1f}", (w // 2 - 15, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 65), 1, cv2.LINE_AA)
            
            self.was_3fingers_down = three_fingers_down
            
            if is_pinching_now and pinch_pos:
                if two_pinches:
                    cx = (pinch_positions[0][0] + pinch_positions[1][0]) // 2
                    cy = (pinch_positions[0][1] + pinch_positions[1][1]) // 2
                    dist = ((pinch_positions[0][0] - pinch_positions[1][0])**2 + 
                            (pinch_positions[0][1] - pinch_positions[1][1])**2) ** 0.5
                    
                    if not self.is_zooming:
                        idx = self.find_photo_at_pos(cx, cy)
                        if idx >= 0:
                            self.is_zooming = True
                            self.zoom_photo_idx = idx
                            self.zoom_initial_dist = dist
                            self.zoom_initial_scale = self.photos[idx].get('scale', 1.0)
                    else:
                        if self.zoom_photo_idx >= 0 and self.zoom_initial_dist > 0:
                            ratio = dist / self.zoom_initial_dist
                            new_scale = self.zoom_initial_scale * ratio
                            new_scale = max(0.3, min(3.0, new_scale))
                            photo = self.photos[self.zoom_photo_idx]
                            old_w, old_h = photo['image'].shape[1], photo['image'].shape[0]
                            photo['scale'] = new_scale
                            photo['size'] = (int(old_w * new_scale), int(old_h * new_scale))
                else:
                    if not self.is_grabbing:
                        idx = self.find_photo_at_pos(pinch_pos[0], pinch_pos[1])
                        if idx >= 0:
                            self.grabbed_photo_idx = idx
                            photo = self.photos[idx]
                            self.grab_offset[0] = pinch_pos[0] - photo['pos'][0]
                            self.grab_offset[1] = pinch_pos[1] - photo['pos'][1]
                            self.smooth_pos[0] = pinch_pos[0]
                            self.smooth_pos[1] = pinch_pos[1]
                            self.is_grabbing = True
                    
                    if self.is_grabbing and self.grabbed_photo_idx >= 0:
                        photo = self.photos[self.grabbed_photo_idx]
                        target_x = pinch_pos[0] - self.grab_offset[0]
                        target_y = pinch_pos[1] - self.grab_offset[1]
                        self.smooth_pos[0] = self.smooth_pos[0] * 0.3 + target_x * 0.7
                        self.smooth_pos[1] = self.smooth_pos[1] * 0.3 + target_y * 0.7
                        photo['pos'][0] = int(self.smooth_pos[0])
                        photo['pos'][1] = int(self.smooth_pos[1])
            elif not is_pinching_now:
                self.is_grabbing = False
                self.grabbed_photo_idx = -1
                self.is_zooming = False
                self.zoom_photo_idx = -1
            
            if self.flash_effect > 0:
                overlay = output.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
                cv2.addWeighted(overlay, self.flash_effect * 0.5, output, 1 - self.flash_effect * 0.5, 0, output)
                self.flash_effect -= 0.05
            
            alive = []
            for p in self.flash_particles:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['life'] -= 0.03
                if p['life'] > 0:
                    alpha = int(p['life'] * 255)
                    cv2.circle(output, (int(p['x']), int(p['y'])), p['size'], (alpha, alpha, alpha), -1)
                    alive.append(p)
            self.flash_particles = alive
            
            for photo in self.photos:
                self.draw_photo(output, photo)
                    
            img_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk
            
        self.root.after(1, self.process_frame)
        
    def add_photo(self, img, shape_pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shape_pts], 255)
        
        photo = img.copy()
        photo[mask == 0] = [10, 10, 10]
        
        pts_array = shape_pts.reshape(-1, 2)
        x_min, y_min = pts_array.min(axis=0)
        x_max, y_max = pts_array.max(axis=0)
        
        crop = photo[y_min:y_max, x_min:x_max].copy()
        
        new_photo = {
            'image': crop,
            'shape_pts': shape_pts - np.array([x_min, y_min]),
            'pos': [int(x_min), int(y_min)],
            'size': (int(x_max - x_min), int(y_max - y_min)),
            'scale': 1.0
        }
        
        self.photos.append(new_photo)
        
    def draw_photo(self, img, photo):
        x = int(photo['pos'][0])
        y = int(photo['pos'][1])
        pw, ph = photo['size']
        scale = photo.get('scale', 1.0)
        
        h_img, w_img, _ = img.shape
        
        if scale != 1.0:
            orig_h, orig_w = photo['image'].shape[:2]
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized = cv2.resize(photo['image'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_shape = (photo['shape_pts'] * scale).astype(np.int32)
        else:
            resized = photo['image']
            resized_shape = photo['shape_pts'].astype(np.int32)
        
        rw, rh = resized.shape[1], resized.shape[0]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + rw)
        y2 = min(h_img, y + rh)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        region = resized[src_y1:src_y2, src_x1:src_x2]
        
        if region.shape[0] > 0 and region.shape[1] > 0:
            mask = np.zeros(region.shape[:2], dtype=np.uint8)
            shifted_pts = resized_shape - np.array([src_x1, src_y1])
            cv2.fillPoly(mask, [shifted_pts], 255)
            
            idx = mask > 0
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            tinted = np.stack([gray_region // 3, gray_region, gray_region // 3], axis=-1).astype(np.uint8)
            
            img[y1:y2, x1:x2][idx] = tinted[idx]
        
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
