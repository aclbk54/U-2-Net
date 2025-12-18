import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Подключаем модель
try:
    from model.u2net import U2NET
except ImportError:
    # Фоллбек для локального запуска, если PYTHONPATH не настроен
    sys.path.append(os.path.join(os.getcwd(), 'u2net_repo'))
    from model.u2net import U2NET

def load_model(model_path):
    net = U2NET(3, 1)
    # Всегда используем CPU для максимальной повторяемости результата
    device = torch.device('cpu') 
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    return net

def preprocess(image):
    # Стандартный препроцессинг для U-2-Net
    transformer = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    return transformer(image).unsqueeze(0)

def normalize(pred):
    ma = torch.max(pred)
    mi = torch.min(pred)
    pred = (pred - mi) / (ma - mi)
    return pred

def compare_images(img_path_a, img_path_b):
    """Сравнивает два изображения и возвращает процент совпадения"""
    img_a = np.array(Image.open(img_path_a).convert('L'))
    img_b = np.array(Image.open(img_path_b).convert('L'))
    
    if img_a.shape != img_b.shape:
        return 0.0, "Different sizes"
    
    # Среднеквадратичная ошибка (MSE)
    err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
    err /= float(img_a.shape[0] * img_a.shape[1])
    
    # Если ошибка очень маленькая, считаем, что совпали
    return err, "OK"

def main():
    # Пути
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output')
    ref_dir = os.path.join(base_dir, 'data', 'reference')
    weights_path = os.path.join(base_dir, 'saved_models', 'u2net.pth')

    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка модели
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)
        
    print("[INFO] Loading model on CPU...")
    net = load_model(weights_path)

    # 2. Поиск изображений
    files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    if not files:
        print("[ERROR] No images found in data/input")
        sys.exit(1)

    all_passed = True

    for img_name in files:
        print(f"\nProcessing {img_name}...")
        
        # Генерация
        img_path = os.path.join(input_dir, img_name)
        orig_img = Image.open(img_path).convert('RGB')
        inp = preprocess(orig_img)
        
        with torch.no_grad():
            d1, *_ = net(inp)
        
        pred = d1[:, 0, :, :]
        pred = normalize(pred)
        
        # Сохранение результата
        pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        mask = Image.fromarray(pred_np).resize(orig_img.size, resample=Image.BILINEAR)
        
        output_path = os.path.join(output_dir, img_name)
        mask.save(output_path)
        print(f"[INFO] Generated result: {output_path}")

        # 3. СРАВНЕНИЕ
        ref_path = os.path.join(ref_dir, img_name)
        if os.path.exists(ref_path):
            print(f"[TEST] Comparing with reference: {ref_path}")
            err, msg = compare_images(output_path, ref_path)

            if err < 1.0: 
                print(f"[PASS] Images match! (MSE: {err:.4f})")
            else:
                print(f"[FAIL] Images different! (MSE: {err:.4f})")
                all_passed = False
        else:
            print("[WARN] No reference file found. Cannot verify reproducibility.")
            print("If this is the first run, copy the output to 'data/reference' folder.")

    if all_passed:
        print("\n=== SUCCESS: All reproducibility tests passed ===")
    else:
        print("\n=== FAILURE: Results do not match references ===")
        sys.exit(1)

if __name__ == "__main__":
    main()