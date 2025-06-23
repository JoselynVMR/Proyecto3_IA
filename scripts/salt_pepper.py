import torch

def salt_and_pepper_noise(img, amount=0.02):
    noisy = img.clone()
    c, h, w = noisy.shape

    num_salt = int(amount * h * w)

    coords = [torch.randint(0, i, (num_salt,)) for i in (h, w)]
    for ch in range(c):
        noisy[ch, coords[0], coords[1]] = 1.0

    coords = [torch.randint(0, i, (num_salt,)) for i in (h, w)]
    for ch in range(c):
        noisy[ch, coords[0], coords[1]] = 0.0

    return noisy
