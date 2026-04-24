import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio
import yt_dlp
from torchvision import transforms
from PIL import Image
import warnings
import time

from model import DeepfakeStackingModel

try:
    from moviepy.editor import VideoFileClip
except ModuleNotFoundError:
    from moviepy import VideoFileClip

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 장전 (4개의 하위 모델이 이 안에 모두 캡슐화되어 있음)
ensemble_model = DeepfakeStackingModel().to(device)
try:
    ensemble_model.load_state_dict(torch.load('best_ensemble_model.pth', map_location=device))
    ensemble_model.eval() 
except FileNotFoundError:
    print(" 'best_ensemble_model.pth' 파일을 찾을 수 없습니다.")

def test_deepfake_video(local_path, global_path, freq_path, audio_path):
    print(f" 입력된 4가지(Local, Global, Freq, Audio) 데이터를 분석하는 중입니다...")
    
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 1 & 2. 시각 데이터 (Local & Global) 텐서화
    local_tensor = transform_img(Image.open(local_path).convert('RGB')).unsqueeze(0).to(device)
    global_tensor = transform_img(Image.open(global_path).convert('RGB')).unsqueeze(0).to(device)
    
    # 3. 주파수 데이터 텐서화
    freq_data = np.load(freq_path)
    freq_tensor = torch.from_numpy(freq_data).float().unsqueeze(0).unsqueeze(0)
    freq_tensor = (freq_tensor - freq_tensor.mean()) / (freq_tensor.std() + 1e-9)
    freq_tensor = freq_tensor.to(device)

    # 4. 청각 데이터 텐서화
    target_sr, max_length = 16000, 3 * 16000
    audio_array, sr = sf.read(audio_path)
    waveform = torch.from_numpy(audio_array).float()
    waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.t()
        
    if sr != target_sr: waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if waveform.shape[1] > max_length: waveform = waveform[:, :max_length]
    else: waveform = F.pad(waveform, (0, max_length - waveform.shape[1]))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )(waveform)

    audio_tensor = torchaudio.transforms.AmplitudeToDB()(mel)
    audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-9)
    audio_tensor = audio_tensor.unsqueeze(0).to(device)

    # 4개의 텐서를 앙상블 모델에 투입!
    with torch.no_grad():
        outputs = ensemble_model(local_tensor, global_tensor, freq_tensor, audio_tensor)
        probs = F.softmax(outputs, dim=1)
        fake_prob, real_prob = probs[0][0].item() * 100, probs[0][1].item() * 100
        _, predicted = torch.max(outputs, 1)
        
    print("=" * 50)
    if predicted.item() == 1: print(f" AI 최종 판별: REAL (진짜 영상입니다!)")
    else: print(f" AI 최종 판별: FAKE (딥페이크로 조작되었습니다!)")
    print(f" 확신도: [진짜일 확률 {real_prob:.1f}%] / [가짜일 확률 {fake_prob:.1f}%]")
    print("=" * 50)

def analyze_video_end_to_end(video_path):
    if not os.path.exists(video_path):
        print(f" 에러: '{video_path}' 파일을 찾을 수 없습니다."); return

    print("=" * 50); print(f" [1단계] '{video_path}' 영상 분석 준비 중...")
    
    # 임시 파일명 4개로 분리
    local_path = "temp_local.jpg"
    global_path = "temp_global.jpg"
    freq_path = "temp_freq.npy"
    audio_path = "temp_audio.wav"
    
    cap = cv2.VideoCapture(video_path)

    # 개선: 여러 프레임 분석 후 다수결
    #ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = [0, total_frames // 4, total_frames // 2, total_frames * 3 // 4]

    results = []
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 각 프레임 분석 후 결과 수집
            results.append(analyze_frame(frame))

    # 다수결로 최종 판별
    final_result = max(set(results), key=results.count)

    if ret:
        # A. 전체 화면 저장 (Global)
        cv2.imwrite(global_path, frame)
        
        # B. 얼굴 추출 (Local) - OpenCV 기본 AI 활용
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 개선: 얼굴 미검출 시 사용자에게 알림
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y + h, x:x + w]
            cv2.imwrite(local_path, face_img)
        else:
            print("얼굴이 감지되지 않았습니다. 전체 화면으로 대체합니다.")
            print("판별 신뢰도가 낮을 수 있습니다.")
            cv2.imwrite(local_path, frame)
            
        # C. 주파수 추출 (Frequency)
        f_shift = np.fft.fftshift(np.fft.fft2(cv2.resize(gray, (128, 128))))
        np.save(freq_path, 20 * np.log(np.abs(f_shift) + 1e-8))
    else:
        print(" 에러: 프레임 추출 실패."); cap.release(); return
    cap.release()
    
    # D. 오디오 추출 (Audio)
    try:
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is not None: video_clip.audio.write_audiofile(audio_path, fps=16000, logger=None)
        else: sf.write(audio_path, np.zeros(16000 * 3), 16000) 
        video_clip.close()
    except Exception as e:
        print(f" 오디오 무음 처리: {e}"); sf.write(audio_path, np.zeros(16000 * 3), 16000)

    print(" [2단계] 재료 4분할 완료! 앙상블 분석 요청...\n")
    test_deepfake_video(local_path, global_path, freq_path, audio_path)
    
    print("\n [3단계] 임시 파일 청소 완료.")
    for p in [local_path, global_path, freq_path, audio_path]:
        if os.path.exists(p): os.remove(p)
    print("=" * 50)

def analyze_url_end_to_end(url):
    print("=" * 50); print(f" [0단계] 링크 영상 다운로드 중... \n URL: {url}")
    temp_video_path = "downloaded_temp.mp4"
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': temp_video_path, 
        'quiet': False, 
        'no_warnings': False
    }
    
    try:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        time.sleep(1) 
            
        print(f" 다운로드 완료! (파일 위치: {os.path.abspath(temp_video_path)})\n")
        analyze_video_end_to_end(temp_video_path)
        
    except Exception as e:
        print(f" 에러: 다운로드 실패. \n상세 내용: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        print("=" * 50)
