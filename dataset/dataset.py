import pandas as pd
import os
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from natsort import natsorted

LABEL_MAP = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3,
    "deer": 4, "dog": 5, "frog": 6, "horse": 7,
    "ship": 8, "truck": 9
}

class TrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # 读取 CSV
        self.root_dir = root_dir
        self.transform = transform  #图像处理与数据增强

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + ".png")  # 构造图片的完整名字，使得label与图片匹配
        image = Image.open(img_name).convert("RGB")

        label_str = self.annotations.iloc[idx, 1]  # 读取字符串标签
        label = LABEL_MAP[label_str]  # 映射到整数

        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # 读取所有图片文件路径
        self.image_paths = natsorted(os.listdir(image_dir))  #使用自然排序，使得10.png在2.png之后
        self.image_paths = [os.path.join(image_dir, img) for img in self.image_paths]
        self.transform = transform

        # 创建一个列表存储每个图像的唯一 id，id 可以通过文件名或序号生成
        self.image_ids = [str(idx) for idx in range(len(self.image_paths))]  # 这里我们使用简单的序号作为 id

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # 获取图片路径
        image = Image.open(img_path).convert("RGB")  # 打开图像

        if self.transform:
            image = self.transform(image)  # 预处理

        # 获取图像的唯一 id
        image_id = self.image_ids[idx]

        return image, image_id

