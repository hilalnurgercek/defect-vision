import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import wide_resnet50_2, resnet18

import numpy as np
import pickle
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import cv2
import os

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),    # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (256, 256))           # Resize manually for now
    img = transform(img).unsqueeze(0)           # Add batch dimension
    return img

outputs = []

def hook(module, input, output):
    outputs.append(output)

def load_model(arch='wide_resnet50_2'):
    if arch == 'resnet18':
        model = resnet18(pretrained=True)
        t_d = 448
        d = 100
    else:
        model = wide_resnet50_2(pretrained=True)
        t_d = 1792
        d = 550

    model.to(device)
    model.eval()

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    return model, t_d, d


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def predict(image_path, model, idx, gaussian_model_path):
    img = preprocess_image(image_path).to(device)

    # Clear previous outputs
    global outputs
    outputs = []

    # Forward pass
    with torch.no_grad():
        _ = model(img)

    # Extract features
    test_outputs = OrderedDict([('layer1', outputs[0]), ('layer2', outputs[1]), ('layer3', outputs[2])])

    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()

    # Load the pre-trained Gaussian model
    with open(gaussian_model_path, 'rb') as f:
        train_outputs = pickle.load(f)

    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        cov_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, cov_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # Upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False).squeeze().numpy()

    # Apply Gaussian smoothing
    score_map = gaussian_filter(score_map, sigma=4)

    return score_map

if __name__ == '__main__':
    model, t_d, d = load_model('wide_resnet50_2')
    idx = torch.tensor(sample(range(0, t_d), d))

    # Example usage:
    image_path = 'path_to_your_image.jpg'
    gaussian_model_path = 'path_to_your_trained_model.pkl'  # like 'temp_wide_resnet50_2/train_bottle.pkl'

    anomaly_map = predict(image_path, model, idx, gaussian_model_path)

    # Save or visualize the result
    import matplotlib.pyplot as plt
    plt.imshow(anomaly_map, cmap='jet')
    plt.colorbar()
    plt.show()
