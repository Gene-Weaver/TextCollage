# train.py
# CUDA 12.1 ===> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6 ===> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv12n model architecture
    model = YOLO('yolov12/yolov12n.pt')

    results = model.train(
      data='datasets/PREP_final/PREP_final.yaml', 
      epochs=250,
      imgsz=640,
    #   device='0,1',
      device='0',
      workers=8,
    )

    # The best trained model will be saved automatically in the 'runs/detect/train/weights/' directory.
    print("Training finished successfully.")