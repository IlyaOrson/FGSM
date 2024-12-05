"Check if category names match class names in file"

from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
category_names = weights.meta["categories"]

with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

for i, (category_name, class_name) in enumerate(zip(category_names, classes)):
    if category_name != class_name:
        print(f"{i}: {category_name} - {class_name}")
