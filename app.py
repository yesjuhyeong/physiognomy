import os
import cv2
import csv
import numpy as np
import dlib
from flask import Flask, request, render_template, jsonify, send_from_directory
import base64
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# CSV 저장 파일 경로
CSV_PATH = 'analysis_data.csv'

# dlib 얼굴 detector 및 68점 landmark predictor 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

##############################
# CSV에 분석 결과 쓰는 함수
##############################
def write_analysis_csv(row_data):
    fieldnames = [
        "timestamp",
        # 눈썹
        "lbw_len", "rbw_len", "brow_len_avg",
        "lbw_curve", "rbw_curve", "brow_curve_avg",
        # 눈
        "lw", "lh", "rw", "rh",
        "eye_w", "eye_h", "area_avg_eye", "aspect_ratio_eye",
        # 코
        "nose_line_len", "dist_mid_nose", "mean_side_dist_nose",
        "nose_curve_val", "nose_len", "nose_wid", "nose_ratio",
        # 입
        "mw", "mh", "area_mouth", "mar",
        "lip_thickness_val", "mouth_corner_diff", "mouth_width_ratio",
        # 턱
        "jaw_width", "dist_jaw8",
        # threshold 값들
        "threshold_brow_len_density",
        "threshold_brow_len_long",
        "threshold_brow_len_mid",
        "threshold_brow_curve_high",
        "threshold_brow_curve_mid",
    ]
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

##############################
# 얼굴 분석 함수
##############################
def analyze_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None, "얼굴을 찾을 수 없습니다."

    rect = rects[0]
    shape = predictor(gray, rect)
    lm = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype='int')

    ###########################
    # Helper Functions
    ###########################
    def length(pt1, pt2):
        return float(np.linalg.norm(pt1 - pt2))

    def line_points_distance(p, a, b):
        A = b[1] - a[1]
        B = a[0] - b[0]
        C = b[0]*a[1] - a[0]*b[1]
        dist = abs(A*p[0] + B*p[1] + C) / (np.sqrt(A*A + B*B) + 1e-9)
        return dist

    ########### (1) 눈썹 분석 ###########
    def analyze_eyebrow(indices):
        pts = lm[indices]
        brow_len = length(pts[0], pts[-1])
        mid_idx = len(indices) // 2
        mid_pt = pts[mid_idx]
        dist_mid = line_points_distance(mid_pt, pts[0], pts[-1])
        return brow_len, dist_mid

    lbw_len, lbw_curve = analyze_eyebrow(range(17, 22))
    rbw_len, rbw_curve = analyze_eyebrow(range(22, 27))
    brow_len_avg = (lbw_len + rbw_len) / 2.0
    brow_curve_avg = (lbw_curve + rbw_curve) / 2.0

    # 임계값 (Threshold)
    THRESH_BROW_DENSITY = 80
    THRESH_BROW_LEN_LONG = 75
    THRESH_BROW_LEN_MID = 55
    THRESH_BROW_CURVE_HIGH = 10
    THRESH_BROW_CURVE_MID = 5

    if brow_len_avg > THRESH_BROW_DENSITY:
        eyebrow_density = "짙은 눈썹"
        density_mean = "정이 많고 사교적, 이성 문제 잦음"
    else:
        eyebrow_density = "옅은 눈썹"
        density_mean = "정이 적어 때때로 신뢰에 문제가 있을 수 있음"

    if brow_len_avg > THRESH_BROW_LEN_LONG:
        brow_length_type = "긴 눈썹"
        length_mean = "정이 많고 에너지 풍부"
    elif brow_len_avg > THRESH_BROW_LEN_MID:
        brow_length_type = "중간 길이 눈썹"
        length_mean = "무난한 편"
    else:
        brow_length_type = "짧은 눈썹"
        length_mean = "독립적이고 고집 셀 수 있음"

    if brow_curve_avg > THRESH_BROW_CURVE_HIGH:
        brow_curve_type = "아치형(곡률 큼)"
        curve_mean = "온화하며 예술적 감성이 뛰어남"
    elif brow_curve_avg > THRESH_BROW_CURVE_MID:
        brow_curve_type = "약간 아치형"
        curve_mean = "적당히 유연함"
    else:
        brow_curve_type = "일자형 또는 낮은 곡률"
        curve_mean = "직선적이며 추진력이 강함"

    eyebrow_type = f"{eyebrow_density}, {brow_length_type}, {brow_curve_type}"
    eyebrow_mean = f"{density_mean} / {length_mean} / {curve_mean}"

    ########### (2) 눈 분석 ###########
    def analyze_eye(indices):
        pts = lm[indices]
        x, y, w, h = cv2.boundingRect(pts)
        return float(w), float(h)

    lw, lh = analyze_eye(range(36, 42))
    rw, rh = analyze_eye(range(42, 48))
    eye_w = (lw + rw) / 2.0
    eye_h = (lh + rh) / 2.0

    left_area = lw * lh
    right_area = rw * rh
    area_avg_eye = (left_area + right_area) / 2.0

    aspect_ratio_eye = eye_w / eye_h if eye_h > 0 else 0

    if area_avg_eye > 900:
        eyes_size_type = "큰 눈"
    elif area_avg_eye > 500:
        eyes_size_type = "보통 크기"
    else:
        eyes_size_type = "작은 눈"

    if aspect_ratio_eye > 2.0:
        eyes_shape_type = "옆으로 긴 편"
    elif aspect_ratio_eye > 1.2:
        eyes_shape_type = "가로세로 비율 무난"
    else:
        eyes_shape_type = "세로로 좀 긴 편"

    size_mean_dict = {
        "큰 눈": "표현력과 리더십이 뛰어남",
        "보통 크기": "무난한 성격",
        "작은 눈": "신중하고 관찰력이 뛰어남"
    }
    shape_mean_dict = {
        "옆으로 긴 편": "사교적, 개방적",
        "가로세로 비율 무난": "균형 잡힌 인상",
        "세로로 좀 긴 편": "내향적, 신중함"
    }
    eyes_type = f"{eyes_size_type}, {eyes_shape_type}"
    eyes_mean = f"{size_mean_dict[eyes_size_type]}, {shape_mean_dict[eyes_shape_type]}"

    ########### (3) 코 분석 ###########
    def nose_curve_measure(lm):
        a = lm[27]
        b = lm[33]
        mid = lm[30]
        dist_mid_ = line_points_distance(mid, a, b)
        side_pts = [lm[i] for i in [31, 32, 34, 35]]
        side_dists = [line_points_distance(pt, a, b) for pt in side_pts]
        mean_side_ = np.mean(side_dists)
        total_ = dist_mid_ + mean_side_
        return dist_mid_, mean_side_, total_

    dist_mid, mean_side_dist, nose_curve_val = nose_curve_measure(lm)

    if nose_curve_val > 15:
        nose_curv_type = "매부리코(곡률 큼)"
        nose_curv_mean = "개성이 강하고 자기주장이 뚜렷함"
    elif nose_curve_val > 7:
        nose_curv_type = "약간 굴곡 코"
        nose_curv_mean = "적당한 개성을 가짐"
    else:
        nose_curv_type = "직선 코"
        nose_curv_mean = "무난함"

    nose_line_len = length(lm[27], lm[33])
    nose_wid = length(lm[31], lm[35])
    nose_len_ = length(lm[27], lm[33])
    nose_ratio = nose_wid / (nose_len_ + 1e-9)

    if nose_ratio > 0.75:
        nose_width_type = "넓은 편"
        nose_width_mean = "금전 낭비 우려 있음"
    elif nose_ratio > 0.5:
        nose_width_type = "적당"
        nose_width_mean = "무난함"
    else:
        nose_width_type = "날렵"
        nose_width_mean = "절약, 신중함"

    nose_type = f"{nose_curv_type}, {nose_width_type}"
    nose_mean = f"{nose_curv_mean} / {nose_width_mean}"

    ########### (4) 입 분석 (세분화) ###########
    mouth_pts = lm[48:68]
    mx, my, mw, mh = cv2.boundingRect(mouth_pts)
    area_mouth = mw * mh
    mar = mw / mh if mh > 0 else 0

    # 4-1) 기본 크기 분류
    if area_mouth > 2500:
        mouth_size_type = "큰 입"
    elif area_mouth > 1500:
        mouth_size_type = "중간 크기"
    else:
        mouth_size_type = "작은 입"

    # 4-2) 기존 모양: 가로/세로 비율
    if mar > 2.2:
        mouth_shape_type = "가로로 긴 형태"
    elif mar > 1.3:
        mouth_shape_type = "비율 무난"
    else:
        mouth_shape_type = "세로로 두툼"

    # 4-3) 입술 두께: boundingRect 높이(mh)를 기준으로
    if mh < 12:
        lip_thickness_desc = "얇은 입술"
    elif mh > 25:
        lip_thickness_desc = "두툼한 입술"
    else:
        lip_thickness_desc = "보통 입술"

    # 4-4) 입 너비 판단: mw 값 기준, 일정 threshold 초과이면 길게 늘어진 입
    if mw > 80:
        long_mouth_desc = "길게 늘어진 입"
    else:
        long_mouth_desc = "너비 보통"

    # 4-5) 입꼬리 판단: 입의 좌우 모서리 (점 48, 54) 이용, y 좌표 차이
    left_corner = lm[48]
    right_corner = lm[54]
    corner_diff = right_corner[1] - left_corner[1]
    if corner_diff < -3:
        corner_desc = "입꼬리가 올라간 입"
    elif corner_diff > 3:
        corner_desc = "입꼬리가 내려간 입"
    else:
        corner_desc = "수평에 가까움"

    mouth_type = f"{mouth_size_type}, {mouth_shape_type}"
    mouth_mean_size = {
        "큰 입": "말재주 뛰어나고 지출 많음",
        "중간 크기": "무난함",
        "작은 입": "조용하고 내성적"
    }
    mouth_mean_shape = {
        "가로로 긴 형태": "사교적, 구설 주의",
        "비율 무난": "특이성 없음",
        "세로로 두툼": "애정표현 풍부"
    }
    # 최종 입 해석: 기존 해석 + 추가 입술 관련 정보
    mouth_mean = (
        f"{mouth_mean_size[mouth_size_type]}, {mouth_mean_shape[mouth_shape_type]} / "
        f"{lip_thickness_desc} / {long_mouth_desc} / {corner_desc}"
    )

    ########### (5) 턱 분석 ###########
    jaw_0 = lm[0]
    jaw_16 = lm[16]
    jaw_width = length(jaw_0, jaw_16)
    jaw_8 = lm[8]
    dist_jaw8 = line_points_distance(jaw_8, jaw_0, jaw_16)

    if jaw_width > 140:
        jaw_w_type = "매우 넓은 턱"
    elif jaw_width > 120:
        jaw_w_type = "넓은 턱"
    elif jaw_width > 100:
        jaw_w_type = "보통 턱"
    else:
        jaw_w_type = "좁은 턱"

    if dist_jaw8 > 25:
        jaw_curve_type = "각지고 발달"
    elif dist_jaw8 > 10:
        jaw_curve_type = "살짝 곡선"
    else:
        jaw_curve_type = "무난하거나 갸름"

    jaw_type = f"{jaw_w_type}, {jaw_curve_type}"
    jaw_mean = f"폭={jaw_w_type}, 곡률={jaw_curve_type}"
    
    ########### 시각화 ###########
    vis = image.copy()
    for (x, y) in lm:
        cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
    cv2.putText(vis, "Detailed Face Analysis", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    filename = f"analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # cv2.imwrite(save_path, vis)
    _, buff = cv2.imencode('.jpg', vis)
    img_b64 = base64.b64encode(buff).decode('utf-8')

    #### CSV 저장용 데이터 딕셔너리 ####
    csv_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # 눈썹
        "lbw_len": f"{lbw_len:.2f}",
        "rbw_len": f"{rbw_len:.2f}",
        "brow_len_avg": f"{brow_len_avg:.2f}",
        "lbw_curve": f"{lbw_curve:.2f}",
        "rbw_curve": f"{rbw_curve:.2f}",
        "brow_curve_avg": f"{brow_curve_avg:.2f}",
        # 눈
        "lw": f"{lw:.2f}",
        "lh": f"{lh:.2f}",
        "rw": f"{rw:.2f}",
        "rh": f"{rh:.2f}",
        "eye_w": f"{eye_w:.2f}",
        "eye_h": f"{eye_h:.2f}",
        "area_avg_eye": f"{area_avg_eye:.2f}",
        "aspect_ratio_eye": f"{aspect_ratio_eye:.2f}",
        # 코
        "nose_line_len": f"{nose_line_len:.2f}",
        "dist_mid_nose": f"{dist_mid:.2f}",
        "mean_side_dist_nose": f"{mean_side_dist:.2f}",
        "nose_curve_val": f"{nose_curve_val:.2f}",
        "nose_len": f"{nose_len_:.2f}",
        "nose_wid": f"{nose_wid:.2f}",
        "nose_ratio": f"{nose_ratio:.2f}",
        # 입
        "mw": f"{mw:.2f}",
        "mh": f"{mh:.2f}",
        "area_mouth": f"{area_mouth:.2f}",
        "mar": f"{mar:.2f}",
        "lip_thickness_val": f"{mh:.2f}",
        "mouth_corner_diff": f"{corner_diff:.2f}",
        "mouth_width_ratio": f"{mw:.2f}",
        # 턱
        "jaw_width": f"{jaw_width:.2f}",
        "dist_jaw8": f"{dist_jaw8:.2f}",
        # Threshold 값들
        "threshold_brow_len_density": "80",
        "threshold_brow_len_long": "75",
        "threshold_brow_len_mid": "55",
        "threshold_brow_curve_high": "10",
        "threshold_brow_curve_mid": "5",
    }
    # write_analysis_csv(csv_data)

    #### 최종 결과 JSON ####
    result_dict = {
        "eyebrows": f"{eyebrow_type} → {eyebrow_mean}",
        "eyes": f"{eyes_type} → {eyes_mean}",
        "nose": f"{nose_type} → {nose_mean}",
        "mouth": f"{mouth_type} → {mouth_mean}",
        "jaw": f"{jaw_type} → {jaw_mean}",
        "saved_image": filename,
        "vis_image": f"data:image/jpeg;base64,{img_b64}"
    }
    return result_dict, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "이미지가 없습니다."})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."})

    file_data = file.read()
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result, err = analyze_face(image)
    if err:
        return jsonify({"error": err})
    return jsonify(result)

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("[WARNING] shape_predictor_68_face_landmarks.dat not found.")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place in the same directory as this script.")
    app.run(debug=True)
