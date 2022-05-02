from torchvision import transforms
from PIL import Image

image_path = "dataset/train/ants/6240338_93729615ec.jpg"
img = Image.open(image_path)

# transforms的使用
# tensor数据类型
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)

