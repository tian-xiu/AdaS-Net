
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests

# model = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
model = AutoModel.from_pretrained("nvidia/MambaVision-L-1K", trust_remote_code=True)
# model = AutoModel.from_pretrained("nvidia/MambaVision-L2-512-21K", trust_remote_code=True)
# model = AutoModel.from_pretrained("nvidia/MambaVision-L3-512-21K", trust_remote_code=True)
# 

# eval mode for inference   MambaVision-L-1K
model.cuda().eval()

# prepare image for the model
url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
image = Image.open(requests.get(url, stream=True).raw)
input_resolution = (3, 1024, 1024)  # MambaVision supports any input resolutions

transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)
inputs = transform(image).unsqueeze(0).cuda()
# model inference
out_avg_pool, features = model(inputs)
print(model)
print("Size of the averaged pool features:", out_avg_pool.size())  # torch.Size([1, 640])
print("Number of stages in extracted features:", len(features)) # 4 stages
print("Size of extracted features in stage 1:", features[0].size()) # torch.Size([1, 80, 56, 56])
print("Size of extracted features in stage 2:", features[1].size()) # torch.Size([1, 80, 56, 56])
print("Size of extracted features in stage 3:", features[2].size()) # torch.Size([1, 80, 56, 56])
print("Size of extracted features in stage 4:", features[3].size()) # torch.Size([1, 640, 7, 7])

# 1: torch.Size([1, 256, 256, 256])
# 2: torch.Size([1, 512, 128, 128])
# 3: torch.Size([1, 1024, 64, 64])
# 4: torch.Size([1, 2048, 32, 32])

# 1: torch.Size([1, 196, 256, 256])
# 2: torch.Size([1, 392, 128, 128])
# 3: torch.Size([1, 784, 64, 64])
# 4: torch.Size([1, 1568, 32, 32])

#  80 160 320 640


# swin 1536 768 384 192