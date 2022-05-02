from PIL import Image
from torchvision import transforms

# writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/20935278_9190345f6b.jpg")
print(img)

#  ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
# writer.add_image("ToTensor", img_tensor)

# Normalize
# 可以改变图片的rgb值
# output[channel]=(input[channel]-mean[channel])/std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

# Resize
print(img)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)

# RandomCrop 随机裁剪
tran_random = transforms.RandomCrop(256)
img_crop = tran_random(img)

print(img_crop)