import torchvision.transforms as T
from PIL import Image

def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    t = T.Compose([
        T.Resize((256,256)),
        T.ToTensor()
    ])
    return t(image).unsqueeze(0)

def postprocess_tensor(tensor):
    if tensor.dim()==4:
        tensor = tensor.squeeze(0)
 #   tensor = (tensor+1)/2
    tensor = tensor.clamp(0,1)
    return T.ToPILImage()(tensor)