import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    CLASS_NAMES,
    CLASS_TO_IDX
)


# ==========================================
# TRAIN TRANSFORM
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# VALIDATION TRANSFORM
# ==========================================

val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# FIND ALL IMAGES
# ==========================================

image_paths = []
labels = []

for class_name in CLASS_NAMES:

    class_folder = DATA_DIR / class_name

    print("Checking:", class_folder)
    print("Exists:", class_folder.exists())

    if not class_folder.exists():
        continue

    for image_path in class_folder.rglob("*"):

        if image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            image_paths.append(image_path)

            labels.append(
                CLASS_TO_IDX[class_name]
            )


print("================================")
print("Total images:", len(image_paths))
print("================================")







# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

train_paths, val_paths, train_labels, val_labels = train_test_split(

    image_paths,
    labels,

    test_size=VAL_SIZE,

    random_state=RANDOM_STATE,

    stratify=labels
)


print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))


# ==========================================
# CUSTOM DATASET
# ==========================================

class CloudDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths

        self.labels = labels

        self.transform = transform


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = self.labels[index]


        # Open image
        image = Image.open(
            image_path
        )


        # Force 3 channels
        image = image.convert("RGB")


        # Apply transform
        if self.transform:

            image = self.transform(image)


        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ==========================================
# CREATE DATASETS
# ==========================================

train_dataset = CloudDataset(

    train_paths,

    train_labels,

    train_transform
)


val_dataset = CloudDataset(

    val_paths,

    val_labels,

    val_transform
)


# ==========================================
# CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


# ==========================================
# TEST DATA LOADER
# ==========================================

if __name__ == "__main__":

    images, labels = next(
        iter(train_loader)
    )

    print(
        "Image shape:",
        images.shape
    )

    print(
        "Label shape:",
        labels.shape
    )