import torch
import numpy as np
from tqdm import tqdm


class Observable:

    def __init__(self):
        self._record = False
        self.observations = {}

    def do_record(self):
        self._record = True

    def record(self, key, data):
        self.observations[key] = data.detach().cpu().numpy()


class LayerForwardHook:
    def __init__(self):
        self.activations = None

    def __call__(self, module, module_input, module_output):
        self.activations = module_output

    def remove(self):
        self.activations = None


class LayerBackwardHook:
    def __init__(self):
        self.gradients_in = None
        self.gradients_out = None

    def __call__(self, module, module_gradient_input, module_gradient_output):
        self.gradients_in = list(module_gradient_input)
        self.gradients_out = module_gradient_output[0]

    def remove(self):
        self.gradients_in = None
        self.gradients_out = None


def normalize_for_display(img, saturation=0.15, brightness=0.5):

    mean, std = img.mean(), img.std()

    if std == 0:
        std += 1e-6

    zero_mean_std_one = img.sub(mean).div(std)
    normalized = zero_mean_std_one.mul(saturation)
    output_img = normalized.add(brightness).clamp(0.0, 1.0)

    return output_img


def activation_maximization(model, transformer, im_size, first_layer_name, layer_name, index, n_iter, lr, device,
                            random_seed=None):

    model.train()
    f_hook = LayerForwardHook()
    b_hook = LayerBackwardHook()

    layer = model._modules[layer_name]
    layer.register_forward_hook(f_hook)
    first_layer = model._modules[first_layer_name]
    first_layer.register_full_backward_hook(b_hook)
    if random_seed is not None:
        np.random.seed(seed=random_seed)
    input_img = np.random.uniform(0.1, 1, im_size).astype(np.float32)

    input_img = transformer(input_img)
    input_img = input_img.unsqueeze(0)
    input_img = input_img.to(device)
    input_img.requires_grad = True
    pbar = tqdm(range(n_iter), leave=False)
    for _ in pbar:

        model.zero_grad()
        model(input_img)
        # calculate gradients with respect of output of a specific filter
        if isinstance(layer, nn.Linear):
            activation = f_hook.activations[:, index]
            activation.backward()
            # normalize gradients
            b_hook.gradients_in[0] /= torch.sqrt(torch.mean(
                torch.mul(b_hook.gradients_in[0], b_hook.gradients_in[0]))) + 0.00001
            # update image
            input_img = input_img + torch.reshape(b_hook.gradients_in[0],
                                                  shape=(1, im_size[2], im_size[1], im_size[0])) * lr
        else:
            activation = torch.mean(f_hook.activations[:, index, :, :])
            activation.backward()
            # normalize gradients
            b_hook.gradients_in[0] /= torch.sqrt(torch.mean(
                torch.mul(b_hook.gradients_in[0], b_hook.gradients_in[0]))) + 0.00001
            # update image
            input_img = input_img + b_hook.gradients_in[0] * lr

        pbar.set_description(f'layer: {layer_name}, unit: {index:2d}, activation: {int(activation.item()):8d}')

    return input_img