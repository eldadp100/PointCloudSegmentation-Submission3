import torch
from torch.optim import Adam

from try_with_pointnet.load_data import get_data
from try_with_pointnet import networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "./checkpoint.pt"
num_segmentation_classes = 55
# parameters
batch_size = 12
epochs = 10
lr = 0.0002


def define_model(ncf):
    return networks.UNetGNN(ncf, num_segmentation_classes=num_segmentation_classes)


def define_loss():
    return torch.nn.CrossEntropyLoss()


def evaluate(model, loss_fn, dl):
    batch_accuracy = []
    batch_loss = []
    for batch in dl:
        batch = batch.to(device)
        ys_pred = model(batch.x, batch.edge_index, batch.batch)
        # calculate batch loss
        loss = loss_fn(ys_pred, batch.y)
        batch_loss.append(loss)
        # calculate batch accuracy
        acc = sum(ys_pred.argmax(dim=1) == batch.y) / len(ys_pred)
        batch_accuracy.append(acc)
    return sum(batch_loss) / len(batch_loss), sum(batch_accuracy) / len(batch_accuracy)


def train(model, loss_fn, optimizer, train_dl, test_dl, device):
    for i in range(epochs):
        # train:
        for batch in train_dl:
            batch = batch.to(device)
            ys_pred = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(ys_pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model checkpoint
        torch.save(model.state_dict(), checkpoint_path)

        # evaluate:
        epoch_acc, epoch_loss = evaluate(model, loss_fn, test_dl)
        print(f"epoch {i}: acc - {epoch_acc}, loss - {epoch_loss}")


if __name__ == '__main__':
    train_dl, test_dl = get_data(batch_size)
    _model = define_model([3, 32, 64, 128]).to(device)
    _loss_fn = define_loss()
    _optimizer = Adam(_model.parameters(), lr)
    train(_model, _loss_fn, _optimizer, train_dl, test_dl, device=device)
