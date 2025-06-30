# train.py
# CUDA 12.1 ===> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6 ===> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

from ultralytics import YOLO
import os 

'''
Disable P2P Communication
The most frequent cause of this timeout on a single machine is an issue with Peer-to-Peer (P2P) communication over the PCIe bus. 
The easiest way to fix this is to tell NCCL not to use direct P2P and instead route the communication through the CPU's main memory. 
This is slightly slower but far more reliable on many systems.

NCCL_P2P_DISABLE=1 YOLO_DEVICE='0,1' python train_linux.py
'''

if __name__ == '__main__':
    # Load the YOLOv12n model architecture
    device = os.environ.get('YOLO_DEVICE', '0')

    model = YOLO('yolov12/yolov12n.pt')

    results = model.train(
      data='datasets/PREP_final/PREP_final.yaml', 
      epochs=250,
      imgsz=640,
      device=device,
      workers=16,
      name='yolo12n', 
    )

    # The best trained model will be saved automatically in the 'runs/detect/train/weights/' directory.
    print("Training finished successfully.")