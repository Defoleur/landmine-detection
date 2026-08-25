from ultralytics import YOLO
import cv2
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/trained_models_project/landmine_multimodal_oblique_7cls_e150_grouped/weights/best.pt"
IMAGE_FOLDER = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_merged_7cls/test_global/images"
CONF = 0.25

EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']


def gather_images(folder):
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def run_inference(model, img_path):
    results = model.predict(source=img_path, conf=CONF, save=False)
    return results[0]


def draw_result(img_path, result, model):
    annotated = result.plot()
    info = f"{os.path.basename(img_path)} - {len(result.boxes)} detections"
    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return annotated


def print_detections(result, model):
    print(f"\nDetected {len(result.boxes)} objects:")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        print(f" - {class_name} (Conf: {conf:.2f})")


def main():
    images = gather_images(IMAGE_FOLDER)
    if not images:
        print(f"No images found in {IMAGE_FOLDER}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Found {len(images)} images.")
    print("Controls:")
    print(" [D] / [Right Arrow] : Next Image")
    print(" [A] / [Left Arrow]  : Previous Image")
    print(" [Q] / [Esc]         : Quit")

    idx = 0
    while True:
        img_path = images[idx]

        print(f"Running inference on: {img_path}")
        result = run_inference(model, img_path)
        print_detections(result, model)

        display_img = draw_result(img_path, result, model)

        info = f"({idx+1}/{len(images)})"
        cv2.putText(display_img, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
        cv2.putText(display_img, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("YOLO Inference Result", display_img)

        key = cv2.waitKey(0)
        if key in [ord('d'), 83, 32]:
            idx = (idx + 1) % len(images)
        elif key in [ord('a'), 81]:
            idx = (idx - 1) % len(images)
        elif key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
