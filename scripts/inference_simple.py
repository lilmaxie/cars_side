import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import tensorflow_hub as hub
from car_azimuth_predictor.utils.training_tools import (
    horizontal_flip_pose_sin_cos_output,
    tf_acc_pi_6_sin_cos_output,
    tf_mean_absolute_angle_error_sin_cos_output,
    tf_median_absolute_angle_error_sin_cos_output,
    tf_r2_angle_score_sin_cos_output,
    tf_rmse_angle_sin_cos_output,
    # Approach 2
    angle_double_output_loss,
    horizontal_flip_pose_double_sigmoid,
    tf_mean_absolute_angle_error_double_sigmoid,
    tf_median_absolute_angle_error_double_sigmoid,
    tf_r2_angle_score_double_sigmoid,
    tf_rmse_angle_score_double_sigmoid,
    tf_acc_pi_6_double_sigmoid,
    np_get_angle_from_double_sigmoids,
    np_get_angle_from_sin_cos,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_image_to_tensor(image_path: str, size=(224, 224)) -> tf.Tensor:
    """Đọc file ảnh và chuyển sang dạng tensor."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, size)
    return image_tensor


def get_position_from_azimuth(azimuth: float) -> str:    
    # Quay góc về trong phạm vi từ -180 đến 180
    # azimuth = (azimuth + 180) % 360 - 180

    # Kiểm tra và trả về vị trí theo phạm vi đã định
    if -10 <= azimuth <= 10:
        return "front"
    elif 30 <= azimuth <= 60:
        return "front left"
    elif 75 <= azimuth <= 105:
        return "side left"
    elif 120 <= azimuth <= 150:
        return "back left"
    elif 170 >= azimuth <= -170:
        return "back"
    elif -150 <= azimuth <= -120:
        return "back right"
    elif -105 <= azimuth <= -75:
        return "side right"
    elif -60 <= azimuth <= -30:
        return "front right"
    else:
        return "invalid"



def predict_azimuth(model, image_paths, approach="1", units="degrees"):
    """
    Thực hiện predict trên danh sách đường dẫn ảnh.
    Trả về list góc (float) cùng mô tả vị trí.
    """
    # Chọn hàm convert output -> góc
    if approach == "1":
        azimuth_converter = np_get_angle_from_sin_cos
    elif approach == "2":
        azimuth_converter = np_get_angle_from_double_sigmoids
    else:
        raise ValueError("Unknown approach (must be '1' or '2').")

    # Load tất cả ảnh thành tensor
    images = []
    for img_path in image_paths:
        images.append(load_image_to_tensor(img_path))
    batch = tf.stack(images, axis=0)

    # Model predict
    predictions = model.predict(batch)

    # Chuyển vector output -> góc (radians)
    azimuths = azimuth_converter(predictions)

    # Chuyển sang degrees nếu cần
    if units == "degrees":
        azimuths = azimuths / np.pi * 180.0

    # Trả kết quả dưới dạng list float
    results = []
    for azimuth in azimuths:
        position = get_position_from_azimuth(azimuth)
        results.append({"azimuth": float(azimuth), "position": position})  # Chuyển `azimuth` thành `float`

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple Inference Script")
    parser.add_argument("input_path", type=str,
                        help="Đường dẫn tới 1 ảnh hoặc 1 folder ảnh")
    parser.add_argument("--model_path", default="models/model_checkpoint.h5",
                        help="Đường dẫn tới file model .h5")
    parser.add_argument("--approach", default="1", choices=["1", "2"],
                        help="Chế độ tính góc (1=SinCos, 2=DoubleSigmoid)")
    parser.add_argument("--units", default="degrees", choices=["degrees", "radians"],
                        help="Hiển thị góc theo độ hoặc radian")
    parser.add_argument("--output_path", default=None,
                        help="Nếu muốn lưu kết quả JSON ra file, đặt đường dẫn ở đây. Nếu không có, sẽ in ra màn hình")

    args = parser.parse_args()

    # Xác định danh sách ảnh
    if os.path.isdir(args.input_path):
        # input_path là folder
        all_files = sorted(os.listdir(args.input_path))
        # Lọc lấy các file ảnh cơ bản (jpg, png, jpeg, ...)
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = [
            os.path.join(args.input_path, f)
            for f in all_files
            if f.lower().endswith(valid_exts)
        ]
    else:
        # input_path là file ảnh
        image_paths = [args.input_path]

    # Kiểm tra có ảnh không
    if not image_paths:
        print(f"[Warning] Không tìm thấy ảnh trong {args.input_path}")
        sys.exit(1)

    # Load model
    model = load_model(args.model_path, custom_objects={
        "KerasLayer": hub.KerasLayer,
        # Approach 2
        "angle_double_output_loss": angle_double_output_loss,
        "tf_mean_absolute_angle_error_double_sigmoid": tf_mean_absolute_angle_error_double_sigmoid,
        "tf_rmse_angle_score_double_sigmoid": tf_rmse_angle_score_double_sigmoid,
        "tf_r2_angle_score_double_sigmoid": tf_r2_angle_score_double_sigmoid,
        "tf_median_absolute_angle_error_double_sigmoid": tf_median_absolute_angle_error_double_sigmoid,
        "tf_acc_pi_6_double_sigmoid": tf_acc_pi_6_double_sigmoid,
        "horizontal_flip_pose_double_sigmoid": horizontal_flip_pose_double_sigmoid,
        # Approach 1
        "tf_mean_absolute_angle_error_sin_cos_output": tf_mean_absolute_angle_error_sin_cos_output,
        "tf_rmse_angle_sin_cos_output": tf_rmse_angle_sin_cos_output,
        "tf_r2_angle_score_sin_cos_output": tf_r2_angle_score_sin_cos_output,
        "tf_median_absolute_angle_error_sin_cos_output": tf_median_absolute_angle_error_sin_cos_output,
        "tf_acc_pi_6_sin_cos_output": tf_acc_pi_6_sin_cos_output,
        "horizontal_flip_pose_sin_cos_output": horizontal_flip_pose_sin_cos_output,
    })

    # Predict
    predictions = predict_azimuth(model, image_paths, approach=args.approach, units=args.units)

    # Tạo danh sách kết quả
    results = []
    for img_path, prediction in zip(image_paths, predictions):
        results.append({
            "image": os.path.basename(img_path),
            "azimuth": prediction["azimuth"],
            "position": prediction["position"]  # Thêm mô tả vị trí
        })

    # Lưu kết quả ra file JSON hoặc in ra màn hình
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu kết quả vào {args.output_path}")
    else:
        # In ra màn hình
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()