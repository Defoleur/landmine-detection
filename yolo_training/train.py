from pathlib import Path
from ultralytics import YOLO

OUTPUT_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_merged_7cls"
DATA_GLOBAL_YAML = f"{OUTPUT_DIR}/data_global.yaml"
DATA_REAL_ONLY_YAML = f"{OUTPUT_DIR}/data_real_only.yaml"

MODEL_CHECKPOINT = "yolo26l.pt" 
EPOCHS = 150
BATCH = 32
IMGSZ = 640
PATIENCE = 50
DEVICE = 0

PROJECT = f"/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/trained_models_project"
RUN_NAME = "landmine_multimodal_oblique_7cls_e150_grouped"

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
        data=DATA_GLOBAL_YAML,
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
    print("TRAINING FINISHED: EXECUTING DUAL-BENCHMARK EVALUATION")
    print("=" * 60)

    print("\n[1/3] Running Validation on Merged Validation Split...")
    val_results = model.val()
    print(f"--> Val mAP50: {val_results.box.map50:.4f}")
    print(f"--> Val mAP50-95: {val_results.box.map:.4f}")

    print("\n[2/3] Running Benchmark A: Global Multi-Class Test Set...")
    global_test_results = model.val(data=DATA_GLOBAL_YAML, split="test")
    print(
        f"--> Global Test (All 7 Classes) mAP50: {global_test_results.box.map50:.4f}"
    )
    print(
        f"--> Global Test (All 7 Classes) mAP50-95: {global_test_results.box.map:.4f}"
    )

    print(
        "\n[3/3] Running Benchmark B: Strict Real-World Test (Sim-to-Real Gap)..."
    )
    real_test_results = model.val(data=DATA_REAL_ONLY_YAML, split="test")
    print(
        f"--> Strict Real-World Test mAP50: {real_test_results.box.map50:.4f}"
    )
    print(
        f"--> Strict Real-World Test mAP50-95: {real_test_results.box.map:.4f}"
    )

    print("\n[+] Exporting best model checkpoint to ONNX...")
    model.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True)
    print("[+] Export complete. Model ready for downstream inference.")


if __name__ == "__main__":
    main()