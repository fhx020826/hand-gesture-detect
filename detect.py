import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import time

# 初始化MediaPipe Hands和Drawing工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def vector_2d_angle(v1, v2):
    """
    计算两个二维向量v1和v2之间的夹角（0~180度）。
    若计算异常则返回65535.
    """
    v1_x, v1_y = v1
    v2_x, v2_y = v2
    try:
        dot = v1_x * v2_x + v1_y * v2_y
        norm1 = math.sqrt(v1_x**2 + v1_y**2)
        norm2 = math.sqrt(v2_x**2 + v2_y**2)
        angle_ = math.degrees(math.acos(dot / (norm1 * norm2)))
    except:
        angle_ = 65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def vector_3d_angle(v1, v2):
    """
    计算两个三维向量v1和v2之间的夹角（0~180度）。
    若计算异常则返回65535.
    """
    try:
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        angle_ = math.degrees(math.acos(dot / (norm1 * norm2)))
    except:
        angle_ = 65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def hand_angle(hand_landmarks):
    """
    计算手指关节的角度列表：
    返回 [拇指, 食指, 中指, 无名指, 小拇指] 各自的角度。
    每个角度通过对比指根与指尖的方向得到。
    """
    angle_list = []
    # 拇指
    angle = vector_2d_angle(
        ((hand_landmarks[0][0] - hand_landmarks[2][0]), (hand_landmarks[0][1] - hand_landmarks[2][1])),
        ((hand_landmarks[3][0] - hand_landmarks[4][0]), (hand_landmarks[3][1] - hand_landmarks[4][1]))
    )
    angle_list.append(angle)

    # 食指
    angle = vector_2d_angle(
        ((hand_landmarks[0][0] - hand_landmarks[6][0]), (hand_landmarks[0][1] - hand_landmarks[6][1])),
        ((hand_landmarks[7][0] - hand_landmarks[8][0]), (hand_landmarks[7][1] - hand_landmarks[8][1]))
    )
    angle_list.append(angle)

    # 中指
    angle = vector_2d_angle(
        ((hand_landmarks[0][0] - hand_landmarks[10][0]), (hand_landmarks[0][1] - hand_landmarks[10][1])),
        ((hand_landmarks[11][0] - hand_landmarks[12][0]), (hand_landmarks[11][1] - hand_landmarks[12][1]))
    )
    angle_list.append(angle)

    # 无名指
    angle = vector_2d_angle(
        ((hand_landmarks[0][0] - hand_landmarks[14][0]), (hand_landmarks[0][1] - hand_landmarks[14][1])),
        ((hand_landmarks[15][0] - hand_landmarks[16][0]), (hand_landmarks[15][1] - hand_landmarks[16][1]))
    )
    angle_list.append(angle)

    # 小拇指
    angle = vector_2d_angle(
        ((hand_landmarks[0][0] - hand_landmarks[18][0]), (hand_landmarks[0][1] - hand_landmarks[18][1])),
        ((hand_landmarks[19][0] - hand_landmarks[20][0]), (hand_landmarks[19][1] - hand_landmarks[20][1]))
    )
    angle_list.append(angle)

    return angle_list

def h_gesture(angle_list, palm_angle, palm_facing_angle, fingers_pointing_toward_screen, is_all_fingers_open, history_x,
              hand_local, normal, move_thr=30):
    """
    根据手指角度列表及手掌、手指朝向信息、历史位置等，判断手势名称。

    angle_list：手指角度列表
    palm_angle：手掌角度（平面内）
    palm_facing_angle：手掌法线与摄像头方向夹角
    fingers_pointing_toward_screen：手指是否朝向屏幕（z值对比）
    is_all_fingers_open：是否所有手指打开
    history_x：手腕x坐标的历史，用于检测水平移动（如Wave）
    hand_local：当前手的21个关键点坐标(x,y,z)
    normal：手掌法线向量
    move_thr：Wave手势检测移动阈值

    返回：gesture_str（手势名称字符串）
    """
    thr_angle = 65.
    thr_angle_s = 49.
    gesture_str = ""

    def finger_straight(a): return a < thr_angle_s
    def finger_bent(a): return a > thr_angle

    # 手指伸直/弯曲状态判定
    thumb_straight = finger_straight(angle_list[0])
    index_straight = finger_straight(angle_list[1])
    middle_straight = finger_straight(angle_list[2])
    ring_straight = finger_straight(angle_list[3])
    pinky_straight = finger_straight(angle_list[4])

    thumb_bent = finger_bent(angle_list[0])
    index_bent = finger_bent(angle_list[1])
    middle_bent = finger_bent(angle_list[2])
    ring_bent = finger_bent(angle_list[3])
    pinky_bent = finger_bent(angle_list[4])

    # 手部关键点坐标（用于判断相对位置和距离）
    wrist_x, wrist_y, wrist_z = hand_local[0]
    thumb_tip_x, thumb_tip_y, thumb_tip_z = hand_local[4]
    index_tip_x, index_tip_y, index_tip_z = hand_local[8]

    if 65535. not in angle_list:

        # Wave与某些条件下的Handshake、Palm_up、Stop依次判断
        if is_all_fingers_open and len(history_x) >= 2:
            x_diff = abs(history_x[-1] - history_x[-2])
            if x_diff > move_thr:
                # 全张开手指且有明显水平移动 -> Wave
                gesture_str = "Wave"
            else:
                # 无明显移动，根据palm_facing_angle区分Stop和Handshake
                if palm_facing_angle < 5:
                    gesture_str = "Stop"
                else:
                    gesture_str = "Handshake"

        # Heart_single：拇指、食指弯曲状态与关节距离判断
        elif thumb_straight and index_bent and middle_bent and ring_bent and pinky_bent:
            thumb_ip_x, thumb_ip_y = hand_local[3][0], hand_local[3][1]
            index_dip_x, index_dip_y = hand_local[7][0], hand_local[7][1]
            dist = np.linalg.norm(np.array([thumb_ip_x, thumb_ip_y]) - np.array([index_dip_x, index_dip_y]))
            if dist < 50:
                gesture_str = "Heart_single"
            elif thumb_tip_y < wrist_y:
                gesture_str = "like"
            else:
                gesture_str = "dislike"

        # ILY手势判断
        elif thumb_straight and index_straight and pinky_straight and middle_bent and ring_bent:
            gesture_str = "ILY"

        # Insult手势判断
        elif middle_straight and thumb_bent and index_bent and ring_bent and pinky_bent and palm_facing_angle > 90:
            gesture_str = "Insult"

        # one
        elif index_straight and thumb_bent and middle_bent and ring_bent and pinky_bent:
            gesture_str = "one"

        # fist
        elif thumb_bent and index_bent and middle_bent and ring_bent and pinky_bent:
            gesture_str = "fist"

        # peace
        elif index_straight and middle_straight and thumb_bent and ring_bent and pinky_bent:
            gesture_str = "peace"

        # call
        elif thumb_straight and pinky_straight and index_bent and middle_bent and ring_bent:
            gesture_str = "call"

        # ok
        elif index_bent and middle_straight and ring_straight and pinky_straight:
            gesture_str = "ok"

        # four
        elif thumb_bent and index_straight and middle_straight and ring_straight and pinky_straight:
            gesture_str = "four"

        # three
        elif thumb_straight and index_straight and middle_straight and ring_bent and pinky_bent:
            gesture_str = "three"

        # three2
        elif index_straight and middle_straight and ring_straight and thumb_bent and pinky_bent:
            gesture_str = "three2"

    return gesture_str

def detect():
    """
    主函数：
    1. 初始化摄像头并配置参数
    2. 使用MediaPipe Hands对手部进行检测和追踪
    3. 计算手势并在图像中标注对应手势名称和FPS
    4. 按 'q' 键退出
    """
    window_name = 'Hand Gesture Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头，请检查摄像头连接或权限设置。")
            return

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        prev_time = 0
        gesture_history = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取视频帧，结束程序。")
                break

            # 翻转图像，便于与镜像行为匹配
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, (hand_landmarks, hand_handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # 绘制手部关键点和连接线
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 获取手的左右标签
                    hand_label = hand_handedness.classification[0].label

                    # 提取手部关键点坐标
                    hand_local = []
                    for lm in hand_landmarks.landmark:
                        x = int(lm.x * frame.shape[1])
                        y = int(lm.y * frame.shape[0])
                        z = lm.z
                        hand_local.append((x, y, z))

                    if hand_local:
                        # 基于2D坐标计算手指角度
                        hand_local_2d = [(p[0], p[1]) for p in hand_local]
                        angle_list = hand_angle(hand_local_2d)

                        # 计算palm_angle
                        wrist = hand_local[0]
                        middle_finger_base = hand_local[9]
                        vector_palm = (middle_finger_base[0] - wrist[0], middle_finger_base[1] - wrist[1])
                        palm_angle = math.degrees(math.atan2(vector_palm[1], vector_palm[0]))
                        if palm_angle < 0:
                            palm_angle += 360

                        thr_angle_s = 49.
                        is_all_fingers_open = all(a < thr_angle_s for a in angle_list)

                        # 判断手指是否指向屏幕（通过z坐标判断）
                        wrist_3d = np.array(hand_local[0])
                        fingers_z = [hand_local[i][2] for i in [8, 12, 16, 20]]
                        fingers_pointing_toward_screen = all(z < wrist_3d[2] for z in fingers_z)

                        # 计算手掌法线normal，用于判断手掌相对相机方向
                        wrist_3d = np.array(hand_local[0])
                        index_mcp_3d = np.array(hand_local[5])
                        pinky_mcp_3d = np.array(hand_local[17])
                        v1 = index_mcp_3d - wrist_3d
                        v2 = pinky_mcp_3d - wrist_3d
                        normal = np.cross(v1, v2)
                        normal_norm = np.linalg.norm(normal)
                        if normal_norm != 0:
                            normal = normal / normal_norm
                        else:
                            normal = np.array([0,0,1])

                        # palm_facing_angle：手掌法线与相机方向夹角
                        cam_dir = np.array([0,0,-1])
                        palm_facing_angle = vector_3d_angle(normal, cam_dir)

                        # 更新手的历史x坐标用于Wave判定
                        if idx not in gesture_history:
                            gesture_history[idx] = deque(maxlen=2)
                        gesture_history[idx].append(int(wrist[0]))

                        # 获取手势名称
                        gesture_str = h_gesture(
                            angle_list,
                            palm_angle,
                            palm_facing_angle,
                            fingers_pointing_toward_screen,
                            is_all_fingers_open,
                            gesture_history[idx],
                            hand_local,
                            normal,
                            move_thr=20
                        )

                        # 在图像上显示手势名称
                        cv2.putText(frame, f"{hand_label} Hand: {gesture_str}", (int(wrist[0]), int(wrist[1]) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # 计算并显示FPS
            current_time = time.time()
            if prev_time != 0:
                fps = 1 / (current_time - prev_time)
            else:
                fps = 0
            prev_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            # 显示图像窗口
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()
