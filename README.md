<div style="text-align:center;margin-bottom:30px">
<p align="center"><img src="./assets/logo.svg" width="470"/></p>
</div>

<br/>

<p align="center">
In this repository I'm trying to do image segmentations on KITTI Road dataset using UNET built from scratch with PyTorch and serve the trained model over HTTP RESTful API for hosting using Docker containers.
</p>
## Architecture U-NET:
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/43e57521-79f3-48f8-8d1e-9dd9d87d220c)
from : U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

# LOSS function IOU loss :
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/a03fbfba-ae26-4648-ac1e-b581b28c2031)

IOU (Intersection over Union) loss is a valuable tool for optimizing segmentation models. It measures the overlap between predicted and ground truth segmentations, encouraging accurate boundary capture. By minimizing the IOU loss during training, models generate more precise segmentations, improving performance in tasks like medical image analysis and object detection.

Note : binary cross-entropy loss is suitable for pixel-wise classification tasks, while IOU loss is beneficial for evaluating and optimizing segmentation models that require accurate boundary capture. and it made more sense to choose a IOU loss over BCE loss



 
```python
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        # Flatten the input tensors
        y_pred = y_pred.view(-1)        
        y_true = y_true.view(-1)
        # Calculate the confusion matrix
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection

        # Calculate the IoU and return the complement as the loss
        iou = intersection / (union + self.eps)
        return 1 - iou

```
## Results :
![download](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/8b96c367-9b35-43d4-b6a1-ab81db2a164a)
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/4c56e1e1-ea7e-4068-a1bb-9695263fb111)
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/e8884131-84ec-40eb-b542-cd3db3ef3dba)
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/b8f749bd-f6e7-4bb4-b0df-53a831d77af1)
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/cd2e8c3b-432c-45f1-b1b9-62fb69c568f4)
![image](https://github.com/hossamemamo/KITTI-Road-Segmentation-UNET/assets/78453559/816cbba6-5dd2-4ec8-add5-55db6ddce3f1)


Support on Kaggle  : https://www.kaggle.com/code/hossamemamo/kitti-road-segmentation-pytorch-unet-from-scratch
> Disclaimer: This project is still work-in-progress

## Running Server Locally
> You will need `Docker` installed for this to work.

* Clone this repo and unzip the `best_model_state.zip` file (this contains the trained parameters).

* Take the output file `best_model_state.bin` and place it inside the `www/` directory.

* `cd` into the `www/` directory and run `docker build -t torch-server .`

* After it finishes, run `docker run -d -p 5000:5000 torch-server`

* Done! you have the model exposed over `localhost:5000`.

* Send your requests containing the image in the request form data with the name `file` using (postman/insomnia/curl).

## Sending Requests

using `curl`:

```shell
curl -X POST -H "Content-Type: multipart/form-data" -F file="@my_img_file.png" "localhost:5000/upload" -o prediction_image.jpg
```

## Cautions

> follow this for consistent behavior as this software is still in early development.

* your input image to the server should be a `png`.
* the output server prediction image will be `jpg`.

## Development Plan

- [x] Download dataset and preprocess it.
- [x] Built `UNET` architecture and train using PyTorch on GPU.
- [x] Expose the model over an HTTP API.
- [x] Build the server docker image.
- [ ] Deploy the docker image to `docker hub`.
- [ ] Build web front-end interface.
- [ ] Host containers on a cloud provider acting as backend for the front-end user interface.
- [ ] Add support for Real-Time.

## Documentation Todos
- [ ] Add documentation for running the server locally without using `Docker`.

## License
MIT
