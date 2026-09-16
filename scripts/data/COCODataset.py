import json
import torch
from pathlib import Path
from PIL import Image as PILImage
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import v2
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)



class COCOSegmentDataset(Dataset):
    """
    用于加载 COCO 格式分割数据的数据集类。

    该类负责读取 COCO JSON 标注文件，加载图像，并将标注（边界框和多边形/RLE 掩码）
    转换为 SAM3 模型训练所需的格式。

    属性:
        data_dir (Path): 包含训练/验证/测试文件夹的根目录。
        split (str): 数据拆分类型 ('train', 'valid', 'test')。
        resolution (int): 图像缩放的目标分辨率，默认为 1008。
    """

    def __init__(self, data_dir: str, split: str = "train"):
        """
        初始化数据集。

        参数:
            data_dir (str): 根目录路径。
            split (str): 'train', 'valid' 或 'test'。

        异常:
            FileNotFoundError: 如果在指定路径找不到 COCO 标注文件。
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # 加载 COCO 标注
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"未找到 COCO 标注文件: {ann_file}")

        with open(ann_file) as f:
            self.coco_data = json.load(f)

        # 构建索引: image_id -> 图像信息
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.image_ids = sorted(list(self.images.keys()))

        # 构建索引: image_id -> 标注列表
        self.img_to_anns = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # 加载类别信息
        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }
        print(f"已加载 COCO 数据集: {split} 分组")
        print(f"  图像总数: {len(self.image_ids)}")
        print(f"  标注总数: {len(self.coco_data['annotations'])}")
        print(f"  类别列表: {self.categories}")

        self.resolution = 512
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        """返回数据集中的图像总数。"""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Datapoint:
        """
        获取指定索引的数据点。

        读取图像并处理其关联的所有标注，转换为 SAM3 内部使用的 Datapoint 对象。

        参数:
            idx (int): 索引。

        返回:
            Datapoint: 包含图像数据、对象标注和查询信息的对象。
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # 加载图像
        img_path = self.split_dir / img_info["file_name"]
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # 缩放图像
        pil_image = pil_image.resize(
            (self.resolution, self.resolution), PILImage.BILINEAR
        )

        # 转换为张量
        image_tensor = self.transform(pil_image)

        # 获取该图像的标注
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []

        # 缩放因子
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # 获取边界框 - COCO 格式为 [x, y, width, height]
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # 从 category_id 获取类别名称
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # 转换为归一化的 CxCyWH 格式
            x, y, w, h = bbox_coco
            # print(f"原始 COCO bbox: {bbox_coco} (x, y, w, h)", type(w), type(h))
            cx = x + float(w) / 2.0
            cy = y + float(h) / 2.0

            # 缩放并归一化到 [0, 1]
            box_tensor = torch.tensor(
                [
                    cx * scale_w / self.resolution,
                    cy * scale_h / self.resolution,
                    float(w) * scale_w / self.resolution,
                    float(h) * scale_h / self.resolution,
                ],
                dtype=torch.float32,
            )

            # 处理分割掩码 (多边形或 RLE 格式)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    if isinstance(segmentation, dict):
                        # RLE 格式
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # 多边形格式
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        segment = None
                        continue

                    # 缩放到模型分辨率
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t, size=(self.resolution, self.resolution), mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] 布尔张量

                except Exception as e:
                    print(f"警告: 处理图像 {img_id} 掩码时出错: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment,
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor, objects=objects, size=(self.resolution, self.resolution)
        )

        from collections import defaultdict

        # 按类别名称分组对象 ID
        class_to_object_ids = defaultdict(list)
        for obj, class_name in zip(objects, object_class_names):
            class_to_object_ids[class_name.lower()].append(obj.object_id)

        # 每个类别创建一个查询
        queries = []
        if len(class_to_object_ids) > 0:
            for query_text, obj_ids in class_to_object_ids.items():
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1,
                    ),
                )
                queries.append(query)
        else:
            # 无标注时，创建一个通用查询
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1,
                ),
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries, images=[image_obj], raw_images=[pil_image]
        )



class DirectCOCODataset(COCOSegmentDataset):
	'''
	DirectCOCODataset 继承自 COCOSegmentDataset。
	这意味着 DirectCOCODataset 会自动拥有父类定义的所有方法（比如 __len__ 和 __getitem__）。
	潜规则：开发者写 DirectCOCODataset 的目的通常是为了覆盖（Override）父类的 __init__ 方法，
	从而改变读取文件路径的方式，而复用父类加载图像和处理掩码的具体逻辑。
	'''
	def __init__(self, data_dir):
		self.data_dir = Path(data_dir)
		self.split_dir = self.data_dir

		# Load COCO annotations
		ann_file = self.split_dir / "_annotations.coco.json"
		if not ann_file.exists():
			raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

		with open(ann_file, 'r') as f:
			self.coco_data = json.load(f)

		# Build index: image_id -> image info
		self.images = {img['id']: img for img in self.coco_data['images']}
		self.image_ids = sorted(list(self.images.keys()))

		# Build index: image_id -> list of annotations
		self.img_to_anns = {}
		for ann in self.coco_data['annotations']:
			img_id = ann['image_id']
			if img_id not in self.img_to_anns:
				self.img_to_anns[img_id] = []
			self.img_to_anns[img_id].append(ann)

		# Load categories
		self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
		print(f"Loaded COCO dataset from {data_dir}")
		print(f"  Images: {len(self.image_ids)}")
		print(f"  Annotations: {len(self.coco_data['annotations'])}")
		print(f"  Categories: {self.categories}")

		self.resolution = 512
		self.transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		])
