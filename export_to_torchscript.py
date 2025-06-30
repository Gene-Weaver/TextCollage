# export_to_torchscript.py

from ultralytics import YOLO
from openvino import Core
import os

if __name__ == "__main__":
    # Load your custom-trained model
    model = YOLO('./runs/detect/yolo12n/weights/best.pt')

    # Define the path to your data configuration YAML
    data_yaml_path = 'datasets/PREP_final/PREP_final.yaml'

    # --- EXPORT LOGIC ---
    print("\n--- Exporting to TorchScript ---")
    # TorchScript export does not typically need the data config, but it's good practice
    torchscript_path = model.export(format='torchscript')
    print(f"✅ TorchScript model saved to: {torchscript_path}")

    print("\n--- Exporting to OpenVINO with Data Configuration ---")
    # This is the critical change. Provide the data YAML to ensure correct class export.
    openvino_path = model.export(format='openvino', data=data_yaml_path)
    print(f"✅ OpenVINO model saved to: {openvino_path}")

    # --- VERIFICATION ---
    ov_model_xml = os.path.join(openvino_path, "best.xml")
    if not os.path.exists(ov_model_xml):
        # The exporter might name it after the .pt file if it's not 'best'
        ov_model_xml = ov_model_xml.replace('best.xml', model.ckpt_path.split('/')[-1].replace('.pt', '.xml'))

    core = Core()
    ov_model = core.read_model(ov_model_xml)
    print("\n✅ OpenVINO output shape:", ov_model.outputs[0].shape)