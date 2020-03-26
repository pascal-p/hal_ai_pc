"""
Training loop, plot
"""

import torch
# import matplotlib.pylab as plt

# print(torch.cuda.is_available())
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(DEV)

# pylint: disable=invalid-name, too-many-arguments, too-many-locals
def train(model, criterion, train_loader, validation_loader, optimizer,
          len_val_dataset,
          epochs=100, device=DEV):
    """
    Our training loop, using gpu if available, cpu otherwise
    """
    loss_accuracy = {'training_loss': [], 'validation_accuracy': []}

    for _epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            #
            x, y = x.to(device), y.to(device)
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            #
            loss.backward()
            optimizer.step()
            loss_accuracy['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()

        accuracy = 100 * (correct / len_val_dataset)
        loss_accuracy['validation_accuracy'].append(accuracy)

    return loss_accuracy

def plot_(plt, training_parms, key='training_loss'):
    """
    Compare training loss/accuracy on plots
    """
    plt.figure(figsize=(10.1, 8))
    for hsh in training_parms:
        plt.plot(hsh['results'][key], label=hsh['label'])

    plt.ylabel(key)
    plt.xlabel('iteration ')
    plt.title(key + ' iterations')
    plt.legend()
