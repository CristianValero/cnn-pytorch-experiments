import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from models import ModelP4CNNP4, ModelZ2CNN, ModelConvEq2D
from utils import RotMNISTDataset


def load_datasets(rot_mnist):
    global batch_size
    if rot_mnist:
        train_data = torch.utils.data.DataLoader(RotMNISTDataset(dataset='train'), batch_size=batch_size,
                                                   shuffle=True, num_workers=4, pin_memory=True)
    else:
        train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           # torchvision.transforms.Pad(7)
                                       ])),
            batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
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


def train():
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


def evaluate_360deg():
    global test_loader, model, device

    evaluation_history = np.zeros((360, 2))
    for deg in range(-180, 180):
        hits = 0
        total = 0
        # pbar = tqdm(test_loader, colour='CYAN', leave=False)
        with torch.no_grad():
            for x_ts, y_ts in test_loader:
                x_ts = torchvision.transforms.functional.rotate(x_ts, deg, interpolation=InterpolationMode.BILINEAR,
                                                                fill=0)
                output = model(x_ts.to(device))
                y_hat = output.data.max(1, keepdim=True)[1]
                hits += sum(y_hat[:, 0].cpu() == y_ts)
                total += y_ts.shape[0]

        acc = hits / total
        evaluation_history[180 + deg] = [deg, acc]
        log = f'rotation: {deg:3d}, test_acc: {acc:.4f}'
        print(log)
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
    plt.tight_layout()
    plt.savefig(f'./evaluation/eval_360deg_invariance.png', format='png', metadata=None, bbox_inches='tight',
                pad_inches=0.1)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 5
    learning_rate = 0.001
    momentum = 0.5  # Use this parameter with SDG optimizer.
    batch_size = 128

    train_loader, test_loader = load_datasets(rot_mnist=False)

    model = ModelZ2CNN().to(device)
    print_model_info()

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_loss = train()

    print('Evaluating model with rotated images...')
    history = evaluate_360deg()
    plot_evaluation_history(eval_history=history, title='Z2CNN with MNIST')
