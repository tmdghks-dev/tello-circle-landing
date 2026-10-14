from djitellopy import Tello
import cv2
import time
import threading
import numpy as np

# Tello 드론 초기화 및 연결
tello = Tello()
try:
    tello.connect()
    print(f"🔋 배터리: {tello.get_battery()}%")
except Exception as e:
    print(f"❌ Tello 드론 연결 실패: {e}")
    exit()

# 기존 스트림 종료 (혹시 모를 경우 대비)
try:
    tello.streamoff()
except Exception as e:
    print(f"기존 스트림 종료 실패: {e}")
time.sleep(0.5)

camera_stream_initialized = False
frame_read = None

# Tello 카메라 스트림 시작
try:
    tello.streamon()
    print("✅ Tello 카메라 스트림 시작.")
    frame_read = tello.get_frame_read()
    # 카메라 프레임이 수신될 때까지 대기
    for _ in range(50): # 최대 50번 시도 (2.5초)
        if frame_read.frame is not None:
            print("카메라 프레임 수신 시작.")
            camera_stream_initialized = True
            break
        time.sleep(0.05) # 0.05초 대기
    else:
        print("❌ 카메라 스트림 초기화 실패. 프로그램 종료.")
        tello.streamoff()
        exit()
except Exception as e:
    print(f"❌ Tello 카메라 스트림 시작 실패: {e}")
    exit()

# 전역 변수 설정
stop_flight = False # 비행 중단 신호
landed = False      # 착륙 완료 여부
flight_complete = False # 전체 비행 경로 완료 여부

# 비행 속도 및 정렬 관련 상수
FB_SPEED = 12 # 전후 속도
LR_SPEED = 12 # 좌우 속도
UD_SPEED = 12 # 상하 속도
YAW_SPEED = 30 # 회전 속도
FINAL_LAND_PIXEL_TOLERANCE = 90 # 원 중심과의 오차 허용 범위 (픽셀)
KP_LR = 0.05 # 좌우 제어 비례 상수 (초기값)
KP_UD = 0.05 # 상하 제어 비례 상수 (초기값)

# --- 함수 정의 ---

def detect_circle_info(bgr_frame, hsv_frame, target_color_mask, min_rad=10, max_rad=200):
    """
    BGR 프레임과 HSV 프레임에서 특정 색상 마스크를 사용하여 원을 감지합니다.
    Args:
        bgr_frame (numpy.array): 원본 BGR 이미지 프레임.
        hsv_frame (numpy.array): 원본 HSV 이미지 프레임.
        target_color_mask (numpy.array): 감지할 색상의 이진 마스크.
        min_rad (int): 감지할 원의 최소 반지름.
        max_rad (int): 감지할 원의 최대 반지름.
    Returns:
        tuple: 감지된 원의 (중심 x, 중심 y, 반지름) 또는 None.
    """
    # 색상 마스크가 적용된 BGR 이미지 생성
    masked_color_bgr = cv2.bitwise_and(bgr_frame, bgr_frame, mask=target_color_mask)
    # 회색조로 변환
    gray_color = cv2.cvtColor(masked_color_bgr, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 적용하여 노이즈 감소
    blurred_color = cv2.GaussianBlur(gray_color, (9, 9), 2)
    
    # 허프 변환을 사용하여 원 감지
    circles = cv2.HoughCircles(blurred_color, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=80, param2=25, minRadius=min_rad, maxRadius=max_rad)
    
    if circles is not None:
        circles = np.uint16(np.around(circles)) # 원형 값을 정수로 변환
        for i in circles[0, :]:
            cx, cy, radius = i[0], i[1], i[2]
            # 감지된 원이 실제로 해당 색상 마스크와 충분히 겹치는지 확인 (가짜 원 필터링)
            circle_mask = np.zeros_like(target_color_mask, dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), radius, 255, -1)
            overlap = cv2.bitwise_and(target_color_mask, target_color_mask, mask=circle_mask)
            # 원의 넓이의 절반 이상이 색상 마스크와 겹치고, 최소 반지름보다 크면 유효한 원으로 간주
            if np.sum(overlap) / 255 > (np.pi * radius**2 * 0.5) and radius > min_rad:
                return (cx, cy, radius)
    return None

def safe_land():
    """
    드론을 안전하게 착륙시키는 함수.
    """
    global landed
    if landed: # 이미 착륙했다면 다시 시도하지 않음
        return
    print("🛬 안전 착륙 절차 시작...")
    max_attempts = 3 # 최대 착륙 시도 횟수
    for attempt in range(max_attempts):
        try:
            tello.send_rc_control(0, 0, 0, 0) # 모든 조종 값을 0으로 설정하여 정지
            time.sleep(0.5) # 잠시 대기
            tello.land() # 착륙 명령
            time.sleep(0.5) # 착륙 완료까지 대기
            landed = True # 착륙 완료 상태로 변경
            print("✅ 착륙 완료.")
            break
        except Exception as e:
            print(f"❌ 착륙 시도 {attempt+1}/{max_attempts} 실패: {e}")
            if attempt == max_attempts - 1:
                print("⚠️ 모든 착륙 시도 실패.")

def move_to_center_and_land_on_circle():
    """
    드론이 원을 감지하고 화면 중앙에 정렬한 뒤 착륙하는 함수.
    """
    global landed, stop_flight
    frame_w, frame_h = 640, 480 # 프레임 너비, 높이
    cx_frame = frame_w // 2 # 프레임의 중앙 X 좌표
    cy_frame = frame_h // 2 # 프레임의 중앙 Y 좌표
    
    no_circle_detection_count = 0 # 원 미감지 카운터
    max_no_detection_time = 10 # 원 미감지 시 최대 허용 시간 (초)
    start_time = time.time() # 함수 시작 시간 기록
    stable_count = 0 # 중심 정렬 안정성 카운터

    print("🎯 원 중앙 정렬 및 착륙 단계 시작...")

    # 빨간색과 파란색 HSV 범위 정의 (두 가지 색상 감지용)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([90, 50, 50]), np.array([130, 255, 255])

    while not landed and not stop_flight: # 착륙하지 않았고 비행 중단 신호가 없을 때 반복
        # 배터리 부족 또는 원 미감지 시간 초과 시 강제 착륙
        if tello.get_battery() < 20 or (time.time() - start_time > max_no_detection_time and no_circle_detection_count > 50):
            print("⚠️ 배터리 부족 또는 원 감지 실패 지속! 강제 착륙.")
            safe_land()
            break

        # 프레임이 없으면 대기 후 다시 시도
        if frame_read is None or frame_read.frame is None:
            time.sleep(0.05)
            continue

        frame = frame_read.frame
        frame = cv2.flip(frame, 1) # 좌우 반전 (셀카 모드)
        resized = cv2.resize(frame, (frame_w, frame_h)) # 프레임 크기 조정
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV) # BGR을 HSV로 변환

        # 빨간색과 파란색 마스크 생성
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 빨간색 원 먼저 감지 시도
        circle = detect_circle_info(resized, hsv, red_mask)
        color_name = "빨간색"
        if circle is None: # 빨간색 원이 감지되지 않으면 파란색 원 시도
            circle = detect_circle_info(resized, hsv, blue_mask)
            color_name = "파란색"
        
        if circle is None: # 원이 감지되지 않았을 때
            no_circle_detection_count += 1
            if no_circle_detection_count > 50: # 일정 시간 이상 원을 잃으면 나선형 검색 시작
                tello.send_rc_control(0, 0, 0, 0) # 드론 정지
                print("⚠️ 목표 원을 잃었습니다! 나선형 검색 시작.")
                tello.rotate_clockwise(10) # 시계 방향으로 10도 회전
                time.sleep(0.5)
            else: # 잠시 원을 잃었을 때는 정지 상태로 재탐색
                tello.send_rc_control(0, 0, 0, 0)
                print("🔍 원을 잃었습니다. 재탐색 중...")
            time.sleep(0.05)
            continue
        
        # 원이 감지되면 미감지 카운터 초기화
        no_circle_detection_count = 0
        x, y, r = map(int, circle) # 감지된 원의 중심 좌표 (x, y)와 반지름 (r)
        print(f"🟢 {color_name} 원 감지됨: 중심 ({x},{y}), 반지름 {r}")

        # 원의 중심과 프레임 중심 간의 오차 계산
        error_x = x - cx_frame
        error_y = y - cy_frame
        distance_from_center = np.sqrt(error_x**2 + error_y**2) # 유클리드 거리

        if distance_from_center < FINAL_LAND_PIXEL_TOLERANCE: # 원이 충분히 중심에 가까워지면
            stable_count += 1 # 안정성 카운트 증가
            if stable_count >= 5: # 5프레임 이상 안정적으로 중심에 머무르면 착륙 준비
                print(f"✅ 중심 정렬 완료! (거리: {distance_from_center:.1f}px).")
                # 착륙 전 앞으로 38cm 이동
                try:
                    print("➡️ 착륙 전 앞으로 38cm 이동합니다.")
                    tello.move_forward(38) # 38cm 앞으로 이동
                    time.sleep(0.5) # 이동 명령 수행을 위한 대기
                except Exception as e:
                    print(f"❌ 앞으로 이동 중 오류 발생: {e}")
                safe_land() # 안전 착륙 함수 호출
                break # 루프 종료
        else: # 원이 중심에 가깝지 않으면
            stable_count = 0 # 안정성 카운트 초기화
            # 오차 크기에 따라 동적으로 제어 비례 상수 조절
            KP_LR_dynamic = 0.03 if abs(error_x) < 50 else 0.05
            KP_UD_dynamic = 0.03 if abs(error_y) < 50 else 0.05
            
            # 좌우(lr) 및 상하(ud) 속도 계산 및 클리핑
            # error_x가 양수면 원이 오른쪽에 있으므로 드론은 왼쪽으로 (-lr_speed)
            # error_y가 양수면 원이 아래쪽에 있으므로 드론은 위로 (-ud_speed)
            lr_speed = int(np.clip(-error_x * KP_LR_dynamic, -LR_SPEED, LR_SPEED))
            ud_speed = int(np.clip(-error_y * KP_UD_dynamic, -UD_SPEED, UD_SPEED))
            
            print(f"🔄 위치 보정 중 (lr:{lr_speed}, ud:{ud_speed})")
            tello.send_rc_control(lr_speed, 0, ud_speed, 0) # RC 제어 명령 전송
        
        time.sleep(0.05) # 제어 주기 유지를 위한 대기

    if not landed: # 루프가 끝났는데도 착륙하지 않았다면 (예: stop_flight이 True가 된 경우)
        print("🛬 비행 중단 또는 오류로 강제 착륙 시도.")
        safe_land()

def fly_route():
    """
    드론의 사전 정의된 비행 경로를 실행하는 함수.
    """
    global stop_flight, landed, flight_complete
    if not camera_stream_initialized:
        print("❌ 카메라 스트림이 초기화되지 않아 이륙할 수 없습니다.")
        stop_flight = True
        return

    print("🚀 이륙합니다!")
    try:
        tello.takeoff() # 이륙 명령
    except Exception as e:
        print(f"❌ 이륙 실패: {e}")
        stop_flight = True
        return
    start_time = time.time() # 비행 시작 시간 기록 

    # 사전 정의된 비행 경로 액션 목록
    actions = [
        lambda: tello.move_up(39),   
        lambda: tello.move_forward(192), 
        lambda: tello.move_down(20),
        lambda: tello.rotate_clockwise(94), 
        lambda: tello.move_forward(20),
        lambda: tello.move_up(35),
        lambda: tello.move_forward(150)
    ]

    for idx, action in enumerate(actions):
        if stop_flight or landed: # 비행 중단 신호나 착륙이 완료되면 경로 중단
            print("비행 경로 중단됨.")
            safe_land()
            break
        print(f"비행 액션 {idx+1}/{len(actions)} 실행...")
        try:
            action() # 현재 액션 실행
            time.sleep(0.5) # 각 액션 후 잠시 대기
        except Exception as e:
            print(f"❌ 비행 액션 {idx+1} 실패: {e}")
            safe_land() # 오류 발생 시 즉시 착륙
            break

    if not landed and not stop_flight: # 경로가 정상적으로 완료되었고 착륙하지 않았다면
        print("➡️ 비행 경로 완료. 원 감지 및 중앙 정렬 시작.")
        time.sleep(1) # 잠시 대기 후 원 감지 단계 시작
        move_to_center_and_land_on_circle() # 원 감지 및 착륙 함수 호출

    elapsed = time.time() - start_time # 총 비행 시간 계산
    print(f"⏱️ 총 비행 시간: {elapsed:.2f}초")
    flight_complete = True # 비행 완료 상태로 변경

# --- 메인 실행 흐름 ---

# 비행 경로를 별도의 스레드에서 실행하여 영상 처리와 동시 진행
flight_thread = threading.Thread(target=fly_route)
try:
    flight_thread.start() # 비행 스레드 시작
    
    # 메인 스레드에서는 카메라 영상 표시 및 사용자 입력 처리
    while flight_thread.is_alive() or not (landed or stop_flight):
        if frame_read is None or frame_read.frame is None:
            time.sleep(0.1)
            continue

        frame = frame_read.frame
        frame = cv2.flip(frame, 1) # 좌우 반전
        resized = cv2.resize(frame, (640, 480)) # 화면 표시용으로 크기 조정
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # 화면 표시를 위한 색상 마스크 생성
        red_mask_display = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | \
                           cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        blue_mask_display = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        
        display = resized.copy()
        
        # 현재 화면에 감지된 원 정보 표시
        current_circle = detect_circle_info(resized, hsv, red_mask_display)
        if current_circle is None:
            current_circle = detect_circle_info(resized, hsv, blue_mask_display)

        if current_circle:
            x, y, r = map(int, current_circle)
            cv2.circle(display, (x, y), r, (0, 255, 0), 2) # 감지된 원 초록색으로 그리기
            cv2.circle(display, (x, y), 2, (0, 0, 255), 3) # 원의 중심 빨간색 점으로 그리기

        # 화면 중앙과 착륙 허용 범위 표시
        cv2.circle(display, (320, 240), 5, (255, 255, 0), -1) # 화면 중앙 노란색 점
        cv2.circle(display, (320, 240), FINAL_LAND_PIXEL_TOLERANCE, (0, 255, 255), 1) # 착륙 허용 범위 (노란색 원)
        
        cv2.imshow("📷 Tello Camera", display) # 화면에 영상 출력

        key = cv2.waitKey(1) & 0xFF # 키 입력 대기
        if key == ord("q"): # 'q' 키를 누르면 비행 중단
            print("Q 키가 눌렸습니다. 비행 중단 및 착륙 시도.")
            stop_flight = True # 비행 중단 신호 설정
            safe_land() # 안전 착륙 시도
            break
finally:
    # 프로그램 종료 시 항상 실행되는 부분
    if not landed: # 비행이 끝나지 않고 착륙하지 않았다면 강제 착륙
        print(f"⚠️ 착륙 상태: landed={landed}, stop_flight={stop_flight}, 배터리={tello.get_battery()}%")
        safe_land()
    tello.streamoff() # 카메라 스트림 종료
    cv2.destroyAllWindows() # 모든 OpenCV 창 닫기
    try:
        tello.end() # Tello 연결 종료
    except Exception as e:
        print(f"드론 연결 종료 실패: {e}")
    print("✅ 모든 자원 해제 및 프로그램 종료.")