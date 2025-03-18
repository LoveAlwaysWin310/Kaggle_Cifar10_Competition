import os
import torch
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from natsort import natsorted  # 关键：使用自然排序
from tqdm import tqdm  # ✅ 引入 tqdm 进度条

# **CIFAR-10 类别映射**
CIFAR10_LABELS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# **1. 设备选择**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# **2. 加载训练好的模型**
def load_model(model_path):
    model = models.resnet50(weights=None)  # 初始化 ResNet50
    model.fc = nn.Linear(2048, 10)  # 修改最后一层（10 类分类）
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载训练好的参数
    model = model.to(device)
    model.eval()  # 设置推理模式
    return model


# **3. 预处理函数**
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),  # 调整大小
        T.ToTensor(),  # 转换为 Tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    image = Image.open(image_path).convert("RGB")  # 打开图片
    image = transform(image)  # 预处理
    image = image.unsqueeze(0)  # 添加 batch 维度 (1, 3, 224, 224)
    return image.to(device)


# **4. 批量推理文件夹内的所有图片**
def predict_folder(model, folder_path, output_csv="predictions_new.csv"):
    # **使用 natsorted() 进行自然排序**
    image_files = natsorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    predictions = []

    # ✅ 使用 tqdm 进度条
    for img_file in tqdm(image_files, desc="Predicting", unit="img"):
        img_path = os.path.join(folder_path, img_file)

        # **去掉文件扩展名**
        img_name = os.path.splitext(img_file)[0]  # 只保留文件名（不含 .png/.jpg）

        # 预处理 & 推理
        image = preprocess_image(img_path)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)  # 获取最大概率类别索引
            label = CIFAR10_LABELS[predicted.item()]  # 映射到类别名称

        predictions.append((img_name, label))

    # **保存结果到 CSV**
    df = pd.DataFrame(predictions, columns=["id", "label"])
    df.to_csv(output_csv, index=False)
    print(f"✅ 预测结果已保存到 {output_csv}")


# **5. 运行推理**
if __name__ == "__main__":
    model_path = "resnet50_model.pth"  # 你的模型文件
    folder_path = "dataset/test"  # 你的测试图片文件夹

    model = load_model(model_path)  # 加载模型
    predict_folder(model, folder_path)  # 运行推理
