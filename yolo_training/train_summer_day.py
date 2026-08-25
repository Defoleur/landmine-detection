from ultralytics import YOLO

OUTPUT_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_summer_day"
DATA_YAML = f"{OUTPUT_DIR}/data.yaml"

MODEL_CHECKPOINT = "yolo26l.pt" 
EPOCHS = 150
BATCH = 32
IMGSZ = 640
PATIENCE = 50
DEVICE = 0

PROJECT = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/trained_models_project"
RUN_NAME = "landmine_multimodal_oblique_7cls_e150_summer_day"

AUG = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=25.0,
    translate=0.1,
    scale=0.6,
    shear=3.0,
    perspective=0.0008,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.2,
    erasing=0.3,
    bgr=0.0,
)


def main():
    model = YOLO(MODEL_CHECKPOINT)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        project=PROJECT,
        name=RUN_NAME,
        optimizer="auto",
        close_mosaic=15,
        **AUG,
    )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED: EVALUATING SUMMER-DAY MODEL")
    print("=" * 60)

    print("\n[1/2] Standard Validation (Summer-Day Val Split)...")
    val_results = model.val()
    print(f"--> Val mAP50: {val_results.box.map50:.4f}")
    print(f"--> Val mAP50-95: {val_results.box.map:.4f}")

    print("\n[2/2] Test on Summer-Day Test Split...")
    test_results = model.val(data=DATA_YAML, split="test")
    print(f"--> Test mAP50: {test_results.box.map50:.4f}")
    print(f"--> Test mAP50-95: {test_results.box.map:.4f}")

    print("\n[+] Exporting best model checkpoint to ONNX...")
    model.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True)
    print("[+] Export complete.")


if __name__ == "__main__":
    main()