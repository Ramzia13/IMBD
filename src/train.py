import torch
import torch.nn as nn
import torch.optim as optim

from model import DemoGPT
from evaluate import calculate_accuracy
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=3,
    lr=3e-4
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {running_loss/100:.4f}"
                )
                running_loss = 0.0

        val_acc = calculate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.2f}%")
