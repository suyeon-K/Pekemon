import cv2
import matplotlib.pyplot as plt
import torch
from model import Generator

from flask import Flask, request, send_file


app = Flask(__name__)

def load_image(path, size=None):
    image = image2tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    w, h = image.shape[-2:]
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        right = left + crop_size
        top = (h - crop_size) // 2
        bottom = top + crop_size
        image = image[:, :, left:right, top:bottom]

    if size is not None and image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)

    return image


def image2tensor(image):
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.
    return (image - 0.5) / 0.5


def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1, 2, 0).cpu().numpy()
    return tensor * 0.5 + 0.5


def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size, size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

@app.route("/predict3", methods=["POST"])
def process_image1():
    file = request.files['file']
    img_bytes = file.read()
    save_path = "samples/data/input.jpg"
    photo = open(save_path, 'wb')
    photo.write(img_bytes)
    photo.close()

    device = 'cpu'
    torch.set_grad_enabled(False)
    image_size = 300  # Can be tuned, works best when the face width is between 200~250 px

    model = Generator().eval().to(device)
    ckpt = torch.load("checkpoint/generator_celeba_distill.pt", map_location=device)
    model.load_state_dict(ckpt)

    results = []
    # backgroundremover -i save_path -o save_path
    image = load_image(save_path, image_size)

    output = model(image.to(device))

    results.append(output.cpu())
    results = torch.cat(results, 2)

    filename = "samples/data/cat.jpg"
    cv2.imwrite(filename, cv2.cvtColor(255 * tensor2image(results), cv2.COLOR_BGR2RGB))

    return send_file(filename, mimetype='image/gif')

@app.route("/predict1", methods=["POST"])
def process_image2():
    file = request.files['file']
    img_bytes = file.read()
    save_path = "samples/data/input.jpg"
    photo = open(save_path, 'wb')
    photo.write(img_bytes)
    photo.close()

    device = 'cpu'
    torch.set_grad_enabled(False)
    image_size = 300  # Can be tuned, works best when the face width is between 200~250 px

    model = Generator().eval().to(device)

    ckpt = torch.load("checkpoint/face_paint_512_v0.pt", map_location=device)
    model.load_state_dict(ckpt)

    results = []
    # backgroundremover -i save_path -o save_path
    image = load_image(save_path, image_size)

    output = model(image.to(device))

    results.append(output.cpu())
    results = torch.cat(results, 2)

    filename = "samples/data/cat.jpg"
    cv2.imwrite(filename, cv2.cvtColor(255 * tensor2image(results), cv2.COLOR_BGR2RGB))

    return send_file(filename, mimetype='image/gif')

@app.route("/predict2", methods=["POST"])
def process_image3():
    file = request.files['file']
    img_bytes = file.read()
    save_path = "samples/data/input.jpg"
    photo = open(save_path, 'wb')
    photo.write(img_bytes)
    photo.close()

    device = 'cpu'
    torch.set_grad_enabled(False)
    image_size = 300  # Can be tuned, works best when the face width is between 200~250 px

    model = Generator().eval().to(device)

    ckpt = torch.load(f"checkpoint/face_paint_512_v2_0.pt", map_location=device)
    model.load_state_dict(ckpt)

    results = []
    # backgroundremover -i save_path -o save_path
    image = load_image(save_path, image_size)

    output = model(image.to(device))

    results.append(output.cpu())
    results = torch.cat(results, 2)

    filename = "samples/data/cat.jpg"
    cv2.imwrite(filename, cv2.cvtColor(255 * tensor2image(results), cv2.COLOR_BGR2RGB))

    return send_file(filename, mimetype='image/gif')

if __name__ == "__main__":
    app.run(host='0.0.0.0')