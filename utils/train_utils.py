import torch
import torch.optim as optim

def get_optimizer(model, learning_rate=0.001):
    return optim.AdamW(model.parameters(), lr=learning_rate)

def compute_loss(predictions, targets):
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0
    for i in range(len(predictions)):
        loss += criterion(predictions[i].view(-1, predictions[i].size(-1)), targets[i].view(-1))
    return loss
