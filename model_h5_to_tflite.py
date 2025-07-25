import tensorflow as tf
import argparse
import os

# Suppress unessary TensorFlow warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_h5_to_tflite(h5_path, output_dir):
    # 모델 로드
    model = tf.keras.models.load_model(h5_path, compile=False)

    # TFLite 변환기 생성
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 변환 수행
    tflite_model = converter.convert()

    # 저장 경로 설정
    model_name = os.path.splitext(os.path.basename(h5_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")

    # 저장
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to: {tflite_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .h5 Keras model to .tflite")
    parser.add_argument('--h5-path', type=str, default='./models/resnet50.h5', help="Path to the input .h5 model")
    parser.add_argument('--output-dir', type=str, default='./models', help="Directory to save .tflite model")
    args = parser.parse_args()

    convert_h5_to_tflite(args.h5_path, args.output_dir)
