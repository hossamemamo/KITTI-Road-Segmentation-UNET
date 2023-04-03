import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dconv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.left_over = DoubleConv(features[-1], features[-1] * 2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layer in self.downs:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.left_over(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):

            sc = skip_connections[idx // 2]
            x = self.ups[idx](x)

            if x.shape != sc.shape:
                x = TF.resize(x, size=sc.shape[2:])

            concat_skip = torch.cat((sc, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return torch.sigmoid(self.final_conv(x))


# `DEVICE` is cpu for predictions (model already trained).
DEVICE = torch.device("cpu")

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("./best_model_state.bin", map_location=DEVICE))

"""
    API Endpoints
"""
app = FastAPI()


@app.get("/")
def handle_get():
    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    return {"result": str(t.tolist())}


@app.get("/health")
def handle_health_check():
    return {"status": "healthy"}


@app.post("/upload")
async def handle_image(file: UploadFile):
    with open("./my_image.png", "wb") as f:
        f.write(file.file.read())

    image = Image.open("./my_image.png")
    resized_image = image.resize((128, 128))
    img_arr = np.array([(np.asarray(resized_image) / 255.0)]).transpose(0, 3, 1, 2)

    out = (
        model(torch.tensor(img_arr[0][np.newaxis, ...], device=DEVICE).float())
        .cpu()
        .detach()
        .numpy()
    )

    out_img = Image.fromarray(np.uint8(out[0].reshape(128, 128) * 255)).convert("RGB")
    out_img.save("out.jpg")

    return FileResponse("./out.jpg")
