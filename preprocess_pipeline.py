import os
import cv2
import json
import subprocess
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


class FakeCatchPreprocessor:
    def __init__(self, raw_dir="1_Raw_Data", processed_dir="2_Processed_Data"):
        self.raw_dir = raw_dir

        self.spatial_dir = os.path.join(processed_dir, "spatial")
        self.frequency_dir = os.path.join(processed_dir, "frequency")
        self.audio_dir = os.path.join(processed_dir, "audio_visual")

        for d in [self.spatial_dir, self.frequency_dir, self.audio_dir]:
            os.makedirs(os.path.join(d, "real"), exist_ok=True)
            os.makedirs(os.path.join(d, "fake"), exist_ok=True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.to_pil = transforms.ToPILImage()

    def extract_audio(self, video_path, video_name, label):
        audio_out_path = os.path.join(self.audio_dir, label, f"{video_name}.wav")

        # 개선 1: 이미 처리된 파일 스킵 (중복 방지)
        if os.path.exists(audio_out_path):
            return audio_out_path

        command = f'ffmpeg -i "{video_path}" -q:a 0 -map a? "{audio_out_path}" -y -loglevel quiet'
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_out_path

    def process_video_frames(self, video_path, video_name, label, extract_fps=5):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30

        frame_interval = max(int(fps / extract_fps), 1)
        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_interval == 0:

                # 개선 2: 이미 처리된 프레임 스킵
                save_name_img = f"{video_name}_frame{frame_count:04d}.jpg"
                save_name_freq = f"{video_name}_frame{frame_count:04d}.npy"
                img_exists = os.path.exists(os.path.join(self.spatial_dir, label, save_name_img))
                freq_exists = os.path.exists(os.path.join(self.frequency_dir, label, save_name_freq))

                if img_exists and freq_exists:
                    frame_count += 1
                    continue

                # 개선 3: 개별 프레임 에러가 전체를 멈추지 않도록 예외처리
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)

                    face_tensor = self.mtcnn(img_pil)

                    if face_tensor is not None:
                        face_img_pil = self.to_pil(face_tensor / 2 + 0.5)
                        face_img_pil.save(os.path.join(self.spatial_dir, label, save_name_img))

                        face_np = np.array(face_img_pil)
                        face_gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
                        face_gray_resized = cv2.resize(face_gray, (128, 128))
                        face_float32 = np.float32(face_gray_resized) / 255.0

                        h, w = face_float32.shape
                        dct_blocks = np.zeros((h, w), dtype=np.float32)

                        for i in range(0, h, 8):
                            for j in range(0, w, 8):
                                block = face_float32[i:i + 8, j:j + 8]
                                dct_blocks[i:i + 8, j:j + 8] = cv2.dct(block)

                        np.save(os.path.join(self.frequency_dir, label, save_name_freq), dct_blocks)
                        saved_count += 1

                # 개선 3: 에러 발생 시 해당 프레임만 스킵
                except Exception as e:
                    print(f"⚠️ {video_name} frame{frame_count} 처리 중 오류 발생: {e}")

            frame_count += 1

        cap.release()
        return saved_count

    def run(self):
        print(" DFDC 다이렉트 전처리 파이프라인 가동 시작!\n")

        json_paths = []
        for root, dirs, files in os.walk(self.raw_dir):
            if "metadata.json" in files:
                json_paths.append(os.path.join(root, "metadata.json"))

        if not json_paths:
            print(f" {self.raw_dir} 폴더 안에 metadata.json 파일이 없습니다!")
            return

        for json_path in json_paths:
            current_dir = os.path.dirname(json_path)

            # 개선 4: JSON 파일 읽기 에러 처리
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{json_path} JSON 파일 읽기 실패: {e}")
                continue

            for video_filename, info in tqdm(metadata.items(), desc=f"{os.path.basename(current_dir)} 전처리 중"):
                video_path = os.path.join(current_dir, video_filename)

                if not os.path.exists(video_path):
                    continue

                video_name = os.path.splitext(video_filename)[0]
                label = "real" if info["label"] == "REAL" else "fake"

                # 개선 5: 영상 단위 에러가 전체를 멈추지 않도록
                try:
                    self.extract_audio(video_path, video_name, label)
                    self.process_video_frames(video_path, video_name, label)
                except Exception as e:
                    print(f"⚠️ {video_name} 처리 실패: {e}")
                    continue

        print("\n 모든 영상의 3분할 전처리가 완료되었습니다!")


if __name__ == "__main__":
    preprocessor = FakeCatchPreprocessor(raw_dir="0_DFDC_Downloads")
    preprocessor.run()