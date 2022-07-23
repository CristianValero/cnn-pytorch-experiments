import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from surgeon_pytorch import Inspect
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from models import ModelP4CNNP4, ModelZ2CNN, ModelConvEq2D
from plot import Recorder
from utils import RotMNISTDataset
from os.path import exists


def load_datasets(rot_mnist):
    global batch_size
    if rot_mnist:
        print('Using ROT-MNIST dataset to train model...')
        train_data = torch.utils.data.DataLoader(RotMNISTDataset(dataset='train'), batch_size=batch_size,
                                                 shuffle=False, num_workers=4, pin_memory=True)
    else:
        train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           # torchvision.transforms.Pad(7)
                                       ])),
            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                       # torchvision.transforms.Pad(7)
                                   ])),
        batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_data, test_data


def print_model_info():
    global model, device
    print(model)
    print(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params} trainable parameters')


def train(model_name, path):
    global epochs, train_loader, test_loader, model, optimizer, ce_loss

    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        train_pbar = tqdm(train_loader, colour='GREEN', leave=False)
        for x, y in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)

            # forward + backward + optimize
            outputs = model(x)
            loss = ce_loss(outputs, y)
            loss.backward()
            optimizer.step()

            # loss metrics
            running_loss += loss.item()
            count += 1
            train_pbar.set_description(f"loss: {running_loss / count:.6f}")

        epoch_loss = running_loss / count

        total = 0
        hits = 0
        with torch.no_grad():
            model.eval()
            for x_ts, y_ts in tqdm(test_loader, colour='CYAN', leave=False):
                output = model(x_ts.to(device))
                y_hat = output.data.max(1, keepdim=True)[1]
                hits += sum(y_hat[:, 0].cpu() == y_ts)
                total += y_ts.shape[0]

        epoch_acc = hits / total
        log = f'epoch: {epoch + 1:3d}, train_loss: {epoch_loss:.6f}, test_acc: {epoch_acc:.4f}'
        print(log)

        torch.save(model, path)

        loss_history.append(epoch_loss)
    return loss_history


def plot_total_loss(loss, total_epochs, title):
    plt.plot(range(total_epochs), loss)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./evaluation/model_training_loss_e{total_epochs}.png')
    plt.close()


def evaluate_360deg(model_name):
    global test_loader, model, device

    evaluation_history = np.zeros((360, 2))
    model_wrapped = Inspect(model, layer=['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7'])
    recorder_all_layers = Recorder(f'./gifs/model_inspection_{model_name}.gif', fps=2)
    recorder_cnn7 = Recorder(f'./gifs/model_inspection_cnn7_{model_name}.gif', fps=2)
    for deg in range(-180, 180):
        hits = 0
        total = 0
        first_image = True
        with torch.no_grad():
            for x_ts, y_ts in test_loader:
                x_ts = torchvision.transforms.functional.rotate(x_ts, deg, interpolation=InterpolationMode.BILINEAR,
                                                                fill=0)
                output, layers_inspection = model_wrapped(x_ts.to(device))
                layers_inspection.insert(0, x_ts)  # Insert original image to inspection array
                y_hat = output.data.max(1, keepdim=True)[1]
                hits += sum(y_hat[:, 0].cpu() == y_ts)
                total += y_ts.shape[0]

                if first_image:
                    # Plot the inspection of all layers.
                    fig, ax = plt.subplots(3, 8, dpi=150)
                    plt.suptitle(f'Model layers inspection with {model_name}', fontsize=15)
                    for i in range(3):
                        for layer in range(8):
                            ax[i, layer].set_title((f'cnn{layer}', 'input')[layer == 0])
                            # if layer != 0:
                            #     ax[i, layer].imshow(layers_inspection[layer][i, 0][0, :].cpu(), cmap='gray')
                            # else:
                            #     ax[i, layer].imshow(layers_inspection[layer][i, 0].cpu(), cmap='gray')
                            ax[i, layer].imshow(layers_inspection[layer][i, 0].cpu(), cmap='gray')
                            ax[i, layer].set_axis_off()
                    fig.tight_layout()
                    recorder_all_layers.add_frame(fig, save=False)
                    # plt.show()
                    plt.close()

                    # Plot the inspection cnn7 layer with 8x10 map -> 80 channels.
                    layers_inspection[7] = layers_inspection[7].flatten(1, 2)
                    layers_inspection[7] = layers_inspection[7].view(torch.Size([128, 10, 8]))

                    fig, ax = plt.subplots(3, 2, dpi=150)
                    plt.suptitle(f'Output cnn7 from {model_name} model {deg}º', fontsize=15)
                    for i in range(3):
                        ax[i, 0].imshow(layers_inspection[0][i, 0].cpu(), cmap='gray')
                        ax[i, 0].set_axis_off()
                        ax[i, 0].set_title('input')
                        ax[i, 1].imshow(layers_inspection[7][i].cpu(), cmap='gray')
                        ax[i, 1].set_axis_off()
                        ax[i, 1].set_title('cnn7 output')
                    fig.tight_layout()
                    recorder_cnn7.add_frame(fig, save=False)
                    # plt.show()
                    plt.close()

                    first_image = False

        acc = hits / total
        evaluation_history[180 + deg] = [deg, acc]
        log = f'rotation: {deg:3d}, test_acc: {acc:.4f}'
        print(log)

    recorder_all_layers.save_gif()
    recorder_cnn7.save_gif()
    return evaluation_history


def plot_evaluation_history(eval_history, title):
    max_acc = np.max(eval_history[:, 1])
    min_acc = np.min(eval_history[:, 1])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.plot(eval_history[:, 0], eval_history[:, 1])
    ax.axhline(y=max_acc, xmin=0, xmax=359, c='gray', linestyle='--')
    ax.axhline(y=min_acc, xmin=0, xmax=359, c='gray', linestyle='--')
    ax.text(1, max_acc + 0.005, f'{max_acc:.4f}')
    ax.text(1, min_acc - 0.015, f'{min_acc:.4f}')
    ax.set_xlim([-180, 180])
    ax.set_ylim([0, 1])
    ax.set_xticks(range(-180, 181, 45))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel('rotation')
    ax.set_ylabel('accuracy')
    plt.suptitle(title)
    fig.tight_layout()
    plt.savefig(f'./evaluation/eval_360deg_invariance.png', format='png', metadata=None, bbox_inches='tight',
                pad_inches=0.1)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':

    models = {
        1: {
            'name': 'Z2CNN',
            'model': ModelZ2CNN()
        },
        2: {
            'name': 'P4CNNP4',
            'model': ModelP4CNNP4()
        },
        3: {
            'name': 'ConvEq2D',
            'model': ModelConvEq2D()
        },
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 20
    learning_rate = 0.001
    batch_size = 128
    train_model = True
    evaluate_model = True
    model_to_use = 3  # Modify this parameter to switch between the different models.
    use_rot_mnist = False

    train_loader, test_loader = load_datasets(rot_mnist=use_rot_mnist)

    model_path = f'./trained_models/{models[model_to_use]["name"]}.pt'
    if exists(model_path):
        print('A previously trained model has been found. Loading...')
        model = torch.load(model_path)
    else:
        model = models[model_to_use]['model'].to(device)

    print_model_info()

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if train_model:
        print('Preparing to train model...')
        _ = train(model_name=models[model_to_use]["name"], path=model_path)

    if evaluate_model:
        print('Evaluating model with rotated images...')
        history = evaluate_360deg(model_name=models[model_to_use]["name"])
        # plot_evaluation_history(eval_history=history, title=f'{models[model_to_use]["name"]} with '
        #                                                     f'{(f"MNIST", "ROT-MNIST")[use_rot_mnist]}')
