from matplotlib import pyplot as plt
import pandas
import torch
from tqdm import tqdm

from models_finetuning.model_evaluation import eval_model


def train(model, train_dataloader, val_dataloader, batch_size, loss_fn, optimizer, device, scheduler, epochs):
    progress_bar = tqdm(range(len(train_dataloader) * epochs))
    model = model.train()
    history = []
    best_acc = 0

    for epoch_num in range(epochs):
        print("_" * 30)
        print(f'Epoch {epoch_num} started.')

        total_loss = 0
        correct_predictions = 0.0

        for data in train_dataloader:
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(
                device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds ==
                                             torch.argmax(labels, dim=1)).item()

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            # to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

        num_data = len(train_dataloader) * batch_size
        train_acc = correct_predictions / num_data
        train_loss = total_loss / num_data
        print(
            f'Epoch: {epoch_num}, Train Accuracy {train_acc}, Loss:  {train_loss}')

        val_acc, val_loss = eval_model(model, val_dataloader, loss_fn, device)
        print(
            f'Epoch: {epoch_num}, Validation Accuracy {val_acc}, Loss:  {val_loss}')

        history.append({"train_acc": train_acc, "train_loss": train_loss,
                       "val_acc": val_acc, "val_loss": val_loss})

        if val_acc > best_acc:
            torch.save(model.state_dict(), 'best_model.bin')
            best_acc = val_acc

    return history


def plot_performance_history(history: pandas.DataFrame):

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
