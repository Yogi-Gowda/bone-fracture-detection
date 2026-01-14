import os
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from ultralytics import YOLO

# 1. Verify Ultralytics and PyTorch
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"PyTorch version: {torch.__version__}")
print("GPU available:", torch.cuda.is_available())

# Create output directory
os.makedirs('/users/hariprasad/MLExamples/bone-fracture-yolov8/outputs', exist_ok=True)

# 2. Dataset Path Setup
dataset_path = '/users/hariprasad/MLExamples/bone-fracture-yolov8/datasets/bone-fracture-v2-3'
output_path_before =  '/users/hariprasad/MLExamples/bone-fracture-yolov8/outputs_before'
output_path_after = '/users/hariprasad/MLExamples/bone-fracture-yolov8/outputs_after'

# 3. Check Dataset Files
for split in ["train", "valid", "test"]:
    img_path = os.path.join(dataset_path, split, "images")
    label_path = os.path.join(dataset_path, split, "labels")
    if os.path.exists(img_path):
        print(f"{split.capitalize()} images: {len(os.listdir(img_path))} files")
    else:
        print(f"{split.capitalize()} images: MISSING")
    if os.path.exists(label_path):
        print(f"{split.capitalize()} labels: {len(os.listdir(label_path))} files")
    else:
        print(f"{split.capitalize()} labels: MISSING")

# 4. Explore Dataset
def explore_dataset(dataset_path):
    yaml_file = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as file:
            data_yaml = yaml.safe_load(file)
            print("\nDataset config:")
            print(data_yaml)

    for split in ['train', 'valid', 'test']:
        path = os.path.join(dataset_path, split, 'images')
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"{split.capitalize()} images: {count}")


# 5. Train YOLOv8 Model
def train_model(dataset_path, model, epochs=50):
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        patience=15,
	pretrained = True,
	lr0 = 0.025,
	lrf = 1e-3,
        batch=16,
        save=True,
        device='cuda',
        workers=8
    )
    return model, results

# 6. Evaluate Model
def evaluate_model(model, dataset_path,eval_path):
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    val_results = model.val(data=yaml_path, device='cuda')
    print("\nValidation metrics:")
    for k, v in val_results.results_dict.items():
        print(f"{k}: {v}")
   
    metrics_path = val_results.save_dir
    for img_name in ['confusion_matrix.png', 'results.png']:
        img_path = os.path.join(metrics_path, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_name)
            plt.savefig(f'{eval_path}')
            plt.close()

    return val_results

# 7. Test on Images
def test_on_images(model, dataset_path, test_path, num_images=5):
    test_path = os.path.join(dataset_path, 'test/images')
    if not os.path.exists(test_path):
        print("Test images folder not found.")
        return
   
    test_images = os.listdir(test_path)[:num_images]
    for idx, img in enumerate(test_images):
        img_path = os.path.join(test_path, img)
        results = model.predict(img_path, conf=0.25, device='cuda')
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(10, 10))
            plt.imshow(im_array[..., ::-1])
            plt.axis('off')
            plt.title(f'Predictions for {img}')
            plt.savefig(f'{test_path}/test_prediction_{idx+1}.png')
            plt.close()

# 8. Save Model
def save_model(model, output_path):
    os.makedirs(output_path, exist_ok=True)
    pt_path = f"{output_path}/newmodel/finetuned_model.pt"
    model.save(pt_path)
    print(f"Model saved to {pt_path}")
    return pt_path


# 10. Main Execution
if __name__ == '__main__':
    # Explore dataset
    model = YOLO('yolov8n.pt')
    explore_dataset(dataset_path)
    evaluate_model(model,dataset_path,output_path_before)

    # Train the model
    model, results = train_model(dataset_path,model,35)

    # Evaluate the model
    evaluate_model(model, dataset_path,output_path_after)

    # Test on some images
    test_on_images(model, dataset_path,output_path_after)

    # Save the trained model
    model_path = save_model(model,output_path_after)
    print("Model is saved at", model_path)


    
