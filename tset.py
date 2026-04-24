from pipeline import analyze_video_end_to_end, analyze_url_end_to_end

print(" 딥페이크 탐지 시스템 구동을 시작합니다!\n")

# --- 테스트 1: 내 컴퓨터에 있는 영상 판독하기 ---
# analyze_video_end_to_end("zumqqvixhu.mp4")

# --- 테스트 2: 유튜브/쇼츠/SNS 링크 판독하기 ---
# 개선: 실행 실패 시 안내
try:
    analyze_url_end_to_end("https://youtube.com/shorts/KtvK-_mExCI?si=28bLU03rEHd-mLm2")
except Exception as e:
    print(f"분석 실패: {e}")