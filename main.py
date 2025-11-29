"""
AI Driver Monitor — MediaPipe demo (Russian MP3 alerts)
- Анализ закрытия глаз (>2.5s) -> звуковое предупреждение (рус.)
- Анализ отвода взгляда в сторону (>3s) -> звуковое предупреждение (рус.)
- Анализ зевка (рот широко открыт >3s) -> звуковое предупреждение (рус.)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import pygame
import threading

# ---------- Настройки ----------
EAR_CLOSED_THRESHOLD = 0.15   # порог EAR для "закрытых" глаз
EYE_CLOSED_SECONDS = 2.5      # секунды подряд > порога -> предупреждение

GAZE_LEFT_THRESHOLD = -0.15   # порог для взгляда влево (нормализованное смещение)
GAZE_RIGHT_THRESHOLD = 0.15   # порог для взгляда вправо (нормализованное смещение)
GAZE_AWAY_SECONDS = 3.0       # секунды подряд -> предупреждение

MOUTH_OPEN_THRESHOLD = 0.4   # порог открытия рта для зевка
MOUTH_OPEN_SECONDS = 1.5      # секунды подряд -> предупреждение

LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'events.csv')

# Пути к звуковым файлам
SOUND_EYES_CLOSED = "Закрыл_глаза.mp3"
SOUND_GAZE_AWAY = "Внимание_на_дорогу.mp3"
SOUND_YAWNING = "Перерыв.mp3"

# Индексы MediaPipe
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# ---------- Инициализация pygame для воспроизведения звуков ----------
pygame.mixer.init()

# ---------- Вспомогательные функции ----------
def ensure_logfile():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ear', 'perclos', 'gaze_left', 'gaze_right', 'mouth_open'])

def log_event(ear, perclos, gaze_left, gaze_right, mouth_open):
    ensure_logfile()
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), f'{ear:.4f}', f'{perclos:.4f}', int(gaze_left), int(gaze_right), f'{mouth_open:.4f}'])

# ---------- MP3 Player ----------
class MP3Player:
    def __init__(self, cooldown=4.0):
        self.cooldown = cooldown
        self.last_warning_time = {}
        
    def play_sound(self, sound_file, warning_type=""):
        now = time.time()
        
        # Проверка кулдауна для конкретного типа предупреждения
        if warning_type in self.last_warning_time:
            if now - self.last_warning_time[warning_type] < self.cooldown:
                return
        
        # Проверяем существование файла
        if not os.path.exists(sound_file):
            print(f'[SOUND ERROR] Файл не найден: {sound_file}')
            return
            
        print(f'[SOUND] Воспроизведение: {sound_file}')
        
        def play_mp3():
            try:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
                # Ждем окончания воспроизведения
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f'Ошибка воспроизведения звука: {e}')
        
        thread = threading.Thread(target=play_mp3)
        thread.daemon = True
        thread.start()
        
        # Обновляем время последнего предупреждения этого типа
        if warning_type:
            self.last_warning_time[warning_type] = now

# ---------- Geometric helpers ----------
def eye_aspect_ratio(eye_points):
    if len(eye_points) < 6:
        return 0.0
    
    points = np.array(eye_points)
    
    # Вычисляем вертикальные расстояния
    vert1 = np.linalg.norm(points[1] - points[5])
    vert2 = np.linalg.norm(points[2] - points[4])
    
    # Вычисляем горизонтальное расстояние
    horiz = np.linalg.norm(points[0] - points[3])
    
    if horiz == 0:
        return 0.0
        
    return (vert1 + vert2) / (2.0 * horiz)

def get_landmark_point(landmarks, index):
    """Безопасное получение точки по индексу"""
    if index < len(landmarks):
        return landmarks[index]
    return None

# ---------- Улучшенная функция для анализа направления взгляда ----------
def analyze_gaze_direction(landmarks):
    """Анализирует направление взгляда на основе положения зрачков относительно углов глаз"""
    try:
        # Получаем центры зрачков
        left_iris_center = np.mean([landmarks[i] for i in LEFT_IRIS_IDX if i < len(landmarks)], axis=0)
        right_iris_center = np.mean([landmarks[i] for i in RIGHT_IRIS_IDX if i < len(landmarks)], axis=0)
        
        # Получаем углы глаз (внешний и внутренний угол для каждого глаза)
        left_eye_inner = landmarks[LEFT_EYE_IDX[0]]  # Внутренний угол левого глаза
        left_eye_outer = landmarks[LEFT_EYE_IDX[3]]  # Внешний угол левого глаза
        right_eye_inner = landmarks[RIGHT_EYE_IDX[0]]  # Внутренний угол правого глаза
        right_eye_outer = landmarks[RIGHT_EYE_IDX[3]]  # Внешний угол правого глаза
        
        # Вычисляем относительное положение зрачков в пределах глаз
        # Для левого глаза: 0.0 = у внутреннего угла, 1.0 = у внешнего угла
        left_eye_width = left_eye_outer[0] - left_eye_inner[0]
        if left_eye_width != 0:
            left_gaze_ratio = (left_iris_center[0] - left_eye_inner[0]) / left_eye_width
        else:
            left_gaze_ratio = 0.5
            
        # Для правого глаза: 0.0 = у внутреннего угла, 1.0 = у внешнего угла
        right_eye_width = right_eye_outer[0] - right_eye_inner[0]
        if right_eye_width != 0:
            right_gaze_ratio = (right_iris_center[0] - right_eye_inner[0]) / right_eye_width
        else:
            right_gaze_ratio = 0.5
        
        # Нормализуем соотношения (в идеале при прямом взгляде оба значения около 0.5)
        # Преобразуем в диапазон [-0.5, 0.5], где 0 = прямой взгляд
        left_gaze_normalized = left_gaze_ratio - 0.5
        right_gaze_normalized = right_gaze_ratio - 0.5
        
        # Среднее значение для обоих глаз
        avg_gaze_normalized = (left_gaze_normalized + right_gaze_normalized) / 2
        
        # Определяем направление взгляда
        gaze_left = avg_gaze_normalized < GAZE_LEFT_THRESHOLD
        gaze_right = avg_gaze_normalized > GAZE_RIGHT_THRESHOLD
        gaze_away = gaze_left or gaze_right
        
        return gaze_away, gaze_left, gaze_right, avg_gaze_normalized, left_gaze_ratio, right_gaze_ratio
        
    except Exception as e:
        print(f"Gaze analysis error: {e}")
        return False, False, False, 0, 0.5, 0.5

# ---------- Функция для анализа поворота головы ----------
def analyze_head_turn(landmarks, frame_width):
    """Анализирует поворот головы влево/вправо"""
    try:
        # Используем ключевые точки лица для определения поворота
        face_left = 234   # Левая сторона лица
        face_right = 454  # Правая сторона лица
        nose_tip = 1      # Кончик носа
        
        left_side = get_landmark_point(landmarks, face_left)
        right_side = get_landmark_point(landmarks, face_right)
        nose_point = get_landmark_point(landmarks, nose_tip)
        
        if not all([left_side, right_side, nose_point]):
            return False
        
        # Вычисляем центр лица
        face_center_x = (left_side[0] + right_side[0]) / 2
        
        # Вычисляем смещение носа относительно центра лица
        nose_offset = nose_point[0] - face_center_x
        
        # Нормализуем относительно ширины лица
        face_width = abs(right_side[0] - left_side[0])
        if face_width == 0:
            return False
            
        head_turn_ratio = abs(nose_offset) / face_width
        
        # Определяем поворот головы (более строгий порог)
        return head_turn_ratio > 0.25
        
    except Exception as e:
        print(f"Head turn analysis error: {e}")
        return False

# ---------- Функция для определения прямого положения головы ----------
def analyze_head_straight(landmarks, frame_width):
    """Анализирует, держится ли голова прямо"""
    try:
        face_left = 234   # Левая сторона лица
        face_right = 454  # Правая сторона лица
        nose_tip = 1      # Кончик носа
        forehead = 10     # Лоб
        chin = 152        # Подбородок
        
        left_side = get_landmark_point(landmarks, face_left)
        right_side = get_landmark_point(landmarks, face_right)
        nose_point = get_landmark_point(landmarks, nose_tip)
        forehead_point = get_landmark_point(landmarks, forehead)
        chin_point = get_landmark_point(landmarks, chin)
        
        if not all([left_side, right_side, nose_point, forehead_point, chin_point]):
            return False
        
        # Вычисляем центр лица по горизонтали
        face_center_x = (left_side[0] + right_side[0]) / 2
        
        # Вычисляем смещение носа относительно центра лица
        nose_offset = abs(nose_point[0] - face_center_x)
        
        # Нормализуем относительно ширины лица
        face_width = abs(right_side[0] - left_side[0])
        if face_width == 0:
            return False
            
        head_turn_ratio = nose_offset / face_width
        
        # Проверяем вертикальное выравнивание (лоб-нос-подбородок)
        vertical_alignment = abs((forehead_point[0] + chin_point[0]) / 2 - nose_point[0])
        vertical_ratio = vertical_alignment / face_width
        
        # Голова считается прямой, если:
        # 1. Поворот головы минимальный (меньше 0.1)
        # 2. Вертикальное выравнивание хорошее (меньше 0.05)
        head_straight = (head_turn_ratio < 0.1) and (vertical_ratio < 0.05)
        
        return head_straight
        
    except Exception as e:
        print(f"Head straight analysis error: {e}")
        return False

# ---------- Main loop ----------
def main():
    ensure_logfile()
    mp3_player = MP3Player(cooldown=5.0)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: cannot open camera')
        return

    # Таймеры для отслеживания продолжительности состояний
    eye_closed_start = None
    gaze_left_start = None
    gaze_right_start = None
    gaze_straight_head_start = None  # Таймер для взгляда в сторону при прямой голове
    mouth_open_start = None
    
    # Буфер для PERCLOS (процент закрытых глаз)
    ear_buffer = []
    perclos_window = 5.0  # 5 секунд для расчета PERCLOS
    
    print('Запуск AI Driver Monitor (нажмите ESC для выхода)')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        current_time = time.time()
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Конвертируем landmarks в пиксельные координаты
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Отрисовка сетки лица
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # 1. АНАЛИЗ ЗАКРЫТИЯ ГЛАЗ
            ear = 0.0
            try:
                left_eye = [landmarks[i] for i in LEFT_EYE_IDX if i < len(landmarks)]
                right_eye = [landmarks[i] for i in RIGHT_EYE_IDX if i < len(landmarks)]
                
                if len(left_eye) >= 6 and len(right_eye) >= 6:
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
            except Exception as e:
                print(f"EAR calculation error: {e}")
                ear = 0.0
            
            # Обновляем буфер для PERCLOS
            ear_buffer.append((current_time, ear))
            # Удаляем старые записи
            ear_buffer = [(t, e) for t, e in ear_buffer if current_time - t <= perclos_window]
            
            # Вычисляем PERCLOS
            if ear_buffer:
                closed_frames = sum(1 for t, e in ear_buffer if e < EAR_CLOSED_THRESHOLD)
                perclos = closed_frames / len(ear_buffer)
            else:
                perclos = 0.0
            
            # Проверка длительного закрытия глаз
            if ear < EAR_CLOSED_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time
                else:
                    closed_duration = current_time - eye_closed_start
                    if closed_duration >= EYE_CLOSED_SECONDS:
                        mp3_player.play_sound(SOUND_EYES_CLOSED, "eyes_closed")
                        eye_closed_start = current_time  # Сброс таймера
            else:
                eye_closed_start = None
            
            # 2. АНАЛИЗ ОТВОДА ВЗГЛЯДА В СТОРОНЫ
            gaze_away, gaze_left, gaze_right, gaze_offset, left_gaze_ratio, right_gaze_ratio = analyze_gaze_direction(landmarks)
            head_turn = analyze_head_turn(landmarks, w)
            head_straight = analyze_head_straight(landmarks, w)
            
            # Обработка взгляда влево (при любом положении головы)
            if gaze_left:
                if gaze_left_start is None:
                    gaze_left_start = current_time
                else:
                    gaze_duration = current_time - gaze_left_start
                    if gaze_duration >= GAZE_AWAY_SECONDS:
                        mp3_player.play_sound(SOUND_GAZE_AWAY, "gaze_left")
                        gaze_left_start = current_time  # Сброс таймера
            else:
                gaze_left_start = None
            
            # Обработка взгляда вправо (при любом положении головы)
            if gaze_right:
                if gaze_right_start is None:
                    gaze_right_start = current_time
                else:
                    gaze_duration = current_time - gaze_right_start
                    if gaze_duration >= GAZE_AWAY_SECONDS:
                        mp3_player.play_sound(SOUND_GAZE_AWAY, "gaze_right")
                        gaze_right_start = current_time  # Сброс таймера
            else:
                gaze_right_start = None
            
            # ОСОБАЯ СИТУАЦИЯ: голова прямая, но взгляд отведен в сторону
            if head_straight and gaze_away:
                if gaze_straight_head_start is None:
                    gaze_straight_head_start = current_time
                else:
                    gaze_duration = current_time - gaze_straight_head_start
                    if gaze_duration >= GAZE_AWAY_SECONDS:
                        # Особое предупреждение для этой ситуации
                        mp3_player.play_sound(SOUND_GAZE_AWAY, "gaze_straight_head")
                        gaze_straight_head_start = current_time  # Сброс таймера
            else:
                gaze_straight_head_start = None
            
            # 3. АНАЛИЗ ЗЕВКА (ОТКРЫТИЯ РТА)
            mouth_open_ratio = 0.0
            try:
                top = get_landmark_point(landmarks, MOUTH_TOP)
                bottom = get_landmark_point(landmarks, MOUTH_BOTTOM)
                
                if top and bottom:
                    mouth_height = np.linalg.norm(np.array(bottom) - np.array(top))
                    # Нормализуем относительно расстояния между глазами
                    left_eye_ref = get_landmark_point(landmarks, 33)
                    right_eye_ref = get_landmark_point(landmarks, 263)
                    
                    if left_eye_ref and right_eye_ref:
                        eye_distance = np.linalg.norm(np.array(right_eye_ref) - np.array(left_eye_ref))
                        if eye_distance > 0:
                            mouth_open_ratio = mouth_height / eye_distance
            except Exception as e:
                print(f"Mouth detection error: {e}")
            
            # Проверка зевка
            if mouth_open_ratio > MOUTH_OPEN_THRESHOLD:
                if mouth_open_start is None:
                    mouth_open_start = current_time
                else:
                    mouth_duration = current_time - mouth_open_start
                    if mouth_duration >= MOUTH_OPEN_SECONDS:
                        mp3_player.play_sound(SOUND_YAWNING, "yawning")
                        mouth_open_start = current_time
            else:
                mouth_open_start = None
            
            # ОТОБРАЖЕНИЕ ИНФОРМАЦИИ НА ЭКРАНЕ
            y_offset = 30
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1%}", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth: {mouth_open_ratio:.3f}", (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Gaze offset: {gaze_offset:.3f}", (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if gaze_away else (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze L/R: {left_gaze_ratio:.2f}/{right_gaze_ratio:.2f}", (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if gaze_away else (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze: {'LEFT' if gaze_left else 'RIGHT' if gaze_right else 'CENTER'}", (10, y_offset + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if gaze_away else (0, 255, 0), 2)
            cv2.putText(frame, f"Head turn: {'YES' if head_turn else 'NO'}", (10, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if head_turn else (0, 255, 0), 2)
            cv2.putText(frame, f"Head straight: {'YES' if head_straight else 'NO'}", (10, y_offset + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if head_straight else (0, 0, 255), 2)
            
            # Отображение предупреждений
            warning_y = y_offset + 200
            if eye_closed_start:
                closed_time = current_time - eye_closed_start
                cv2.putText(frame, f"Eyes closed: {closed_time:.1f}s", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                warning_y += 25
            
            if gaze_left_start:
                gaze_time = current_time - gaze_left_start
                cv2.putText(frame, f"Gaze left: {gaze_time:.1f}s", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                warning_y += 25
            
            if gaze_right_start:
                gaze_time = current_time - gaze_right_start
                cv2.putText(frame, f"Gaze right: {gaze_time:.1f}s", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                warning_y += 25
            
            if gaze_straight_head_start:
                gaze_time = current_time - gaze_straight_head_start
                cv2.putText(frame, f"Gaze away (straight head): {gaze_time:.1f}s", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # Красный цвет для особого предупреждения
                warning_y += 25
            
            if mouth_open_start:
                mouth_time = current_time - mouth_open_start
                cv2.putText(frame, f"Yawning: {mouth_time:.1f}s", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Логирование каждую секунду
            if int(current_time) % 1 == 0:  # Раз в секунду
                log_event(ear, perclos, gaze_left, gaze_right, mouth_open_ratio)
                
        else:
            # Лицо не обнаружено
            cv2.putText(frame, 'Face not detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            gaze_left_start = None
            gaze_right_start = None
            gaze_straight_head_start = None
        
        # Отображение кадра
        cv2.imshow('AI Driver Monitor (ESC - exit)', frame)
        
        # Выход по ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()