# train.py
# CUDA 12.1 ===> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6 ===> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
'''
Disable P2P Communication
The most frequent cause of this timeout on a single machine is an issue with Peer-to-Peer (P2P) communication over the PCIe bus. 
The easiest way to fix this is to tell NCCL not to use direct P2P and instead route the communication through the CPU's main memory. 
This is slightly slower but far more reliable on many systems.

NCCL_P2P_DISABLE=1 YOLO_DEVICE='0,1' python train_linux.py

@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@software{yolo12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLOv12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {https://github.com/sunsmarterjie/yolov12},
  license = {AGPL-3.0}
}
'''

from ultralytics import YOLO
import os 

from train_utils import evaluate_and_log
from export_models_all_formats import export_all_formats

DEVICE = "1"

if __name__ == '__main__':
    # Load the YOLOv12n model architecture

    # --- wandb setup ---
    os.environ["WANDB_MODE"] = "online"  # or "offline" to cache locally
    os.environ["WANDB_PROJECT"] = "LM3_BBOXES_YOLO12"
    os.environ["WANDB_NAME"] = "plant_angiosperms_1_yolo12n"

    model = YOLO('yolov12/yolov12n.pt')
    results = model.train(
      data='/datab/Component_Detectors_Training/Plant_Angiosperms_1_txt.yaml', 
      epochs=15,
      batch=96,
      workers=12,
      imgsz=640,
      device=DEVICE, 
      name='plant_angiosperms_1_yolo12n', 

      val=True,
      # memory safety
      cache=False,
      deterministic=True,
      # wandb integration
      exist_ok=True,
    )
    export_all_formats(
        weights="runs/detect/plant_angiosperms_1_yolo12n/weights/best.pt",
        model_name="plant_angiosperms_1_yolo12n",
        version="v1.0.0",
        data_yaml="/datab/Component_Detectors_Training/Plant_Angiosperms_1_txt.yaml",
        imgsz=640,
        device=DEVICE,
    )
    print("Training finished successfully.")


    os.environ["WANDB_NAME"] = "plant_angiosperms_1_yolo12x"
    modelx = YOLO('yolov12/yolov12x.pt')
    resultsx = modelx.train(
      data='/datab/Component_Detectors_Training/Plant_Angiosperms_1_txt.yaml', 
      epochs=15,
      batch=18,
      workers=12,
      imgsz=640,
      device=DEVICE,
      name='plant_angiosperms_1_yolo12x', 
    )
    export_all_formats(
        weights="runs/detect/plant_angiosperms_1_yolo12x/weights/best.pt",
        model_name="plant_angiosperms_1_yolo12x",
        version="v1.0.0",
        data_yaml="/datab/Component_Detectors_Training/Plant_Angiosperms_1_txt.yaml",
        imgsz=640,
        device=DEVICE,
        val=True,
        # memory safety
        cache=False,
        deterministic=True,
        # wandb integration
        exist_ok=True,
    )
    print("Training finished successfully.")



    models = {
        # "yolo12n": model,                          # directly use the trained model object
        # "yolo12x": modelx,
        # can also pass weight paths:
        "yolo12n": "runs/detect/plant_angiosperms_1_yolo12n/weights/best.pt",
        "yolo12x": "runs/detect/plant_angiosperms_1_yolo12x/weights/best.pt",
    }
    data_yamls = {
        "Plant": "/datab/Component_Detectors_Training/Plant_Angiosperms_1_txt.yaml",
    }


    evaluate_and_log(
        models=models,
        data_yamls=data_yamls,
        device=DEVICE,
        imgsz=640,
        out_dir="/datab/Component_Detectors_Training/Training_Stats/eval",
        wandb_project="LM2-YOLO12-Evals",
        extra_config={"host": "ubuntu22", "notes": "test-split evaluation"},
    )