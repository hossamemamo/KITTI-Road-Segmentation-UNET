<div style="text-align:center;margin-bottom:30px">
<p align="center"><img src="./assets/logo.svg" width="470"/></p>
</div>

<br/>

<p align="center">
In this repository I'm trying to do image segmentations on KITTI Road dataset using UNET built from scratch with PyTorch and serve the trained model over HTTP RESTful API for hosting using Docker containers.
</p>

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

## Results
Check notebook

## License
MIT
