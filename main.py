import os
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import ultralytics

# 1. Verify Ultralytics and PyTorch
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"PyTorch version: {torch.__version__}")
print("GPU available:", torch.cuda.is_available())

# 2. Dataset Path Setup
dataset_path = './bone-fracture-yolov8/datasets/bone-fracture-v2-3'

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

    train_images = os.path.join(dataset_path, 'train/images')
    if os.path.exists(train_images):
        sample_images = os.listdir(train_images)[:3]
        print("\nSample training images:")
        for img in sample_images:
            img_path = os.path.join(train_images, img)
            image = Image.open(img_path)
            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.title(img)
            plt.show()

        # Show label of the first image
        label_path = os.path.join(dataset_path, 'train/labels', os.path.splitext(sample_images[0])[0] + '.txt')
        if os.path.exists(label_path):
            print("\nSample label format:")
            with open(label_path, 'r') as f:
                print(f.read())
            print("\nYOLO format: class_id center_x center_y width height (normalized)")

# 5. Train YOLOv8 Model
def train_model(dataset_path, model_size='m', epochs=50):
    model = YOLO(f'yolov8{model_size}.pt')
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        patience=15,
        batch=16,
        save=True,
        device='cuda',
        workers=8
    )
    return model, results

# 6. Evaluate Model
def evaluate_model(model, dataset_path):
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    val_results = model.val(data=yaml_path, device='cuda')
    print("\nValidation metrics:")
    for k, v in val_results.results_dict.items():
        print(f"{k}: {v}")
    
    metrics_path = model.trainer.save_dir
    for img_name in ['confusion_matrix.png', 'results.png']:
        img_path = os.path.join(metrics_path, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_name)
            plt.show()

    return val_results

# 7. Test on Images
def test_on_images(model, dataset_path, num_images=5):
    test_path = os.path.join(dataset_path, 'test/images')
    if not os.path.exists(test_path):
        print("Test images folder not found.")
        return
    
    test_images = os.listdir(test_path)[:num_images]
    for img in test_images:
        img_path = os.path.join(test_path, img)
        results = model.predict(img_path, conf=0.25, device='cuda')
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(10, 10))
            plt.imshow(im_array[..., ::-1])
            plt.axis('off')
            plt.title(f'Predictions for {img}')
            plt.show()

# 8. Save Model
def save_model(model, output_path='./bone_fracture_model'):
    os.makedirs(output_path, exist_ok=True)
    pt_path = f"{output_path}/bone_fracture_model.pt"
    model.save(pt_path)
    print(f"Model saved to {pt_path}")
    return pt_path

# 9. Predict on New Image
def predict_fracture(model_path, image_path, conf_threshold=0.3):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold, device='cuda')
    for r in results:
        im_array = r.plot()
        plt.figure(figsize=(12, 12))
        plt.imshow(im_array[..., ::-1])
        plt.axis('off')
        plt.title('Bone Fracture Prediction')
        plt.show()

        if len(r.boxes) > 0:
            print(f"Found {len(r.boxes)} potential fractures:")
            for i, box in enumerate(r.boxes):
                print(f"  Fracture {i+1}: Confidence = {box.conf.item():.2f}, Coordinates = {box.xyxy.tolist()[0]}")
        else:
            print("No fractures detected.")

    return results

# 10. Main Execution
if __name__ == '__main__':
    # Explore dataset
    explore_dataset(dataset_path)

    # Train the model
    model, results = train_model(dataset_path=dataset_path, model_size='m', epochs=50)

    # Evaluate the model
    val_results = evaluate_model(model, dataset_path)

    # Test on some images
    test_on_images(model, dataset_path)

    # Save the trained model
    model_path = save_model(model)

    # Predict on a new image
    test_image_path = './predictions/fracture_hand.jpg'
    if os.path.exists(test_image_path):
        predict_fracture(model_path, test_image_path)
    else:
        print(f"Prediction image not found at: {test_image_path}")
