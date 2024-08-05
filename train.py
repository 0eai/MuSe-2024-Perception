import os

import numpy as np
import torch
import torch.optim as optim

from eval import evaluate, evaluate_multi_att, evaluate_feature_fusion

import wandb


def train(model, train_loader, optimizer, loss_fn, use_gpu=False, device='cpu'):

    train_loss_list = []

    model.train()
    if use_gpu:
        model.to(device)

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.to(device)
            feature_lens = feature_lens.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        preds,_ = model(features, feature_lens)

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    train_loss = np.mean(train_loss_list)
    return train_loss


def save_model(model, model_folder, id):
    model_file_name = f'model_{id}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file)
    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, regularization=0.0, device='cpu'):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    metrics = {"best_val_loss": best_val_loss,
                 "best_val_score": best_val_score,
                 "epoch": -1,
                 "train_loss": float('inf'),
                 "val_loss": float('inf'),
                 "val_score": -1,}
    
    for epoch in range(1, epochs + 1):
        metrics.update({"epoch": epoch})        ## update epoch
        print(f'Training for Epoch {epoch}...')

        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu, device=device)
        metrics.update({"train_loss": train_loss})  ## update train_loss
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu, device=device)
        metrics.update({"val_loss": val_loss, "val_score": val_score})      ## update val_loss, val_score

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            metrics.update({"best_val_score": best_val_score, "best_val_loss": best_val_loss})      ## update best_val_score, best_val_loss
            best_model_file = save_model(model, model_path, identifier)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        wandb.log(metrics)      ## log metrics

    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_score, best_model_file


##################################################
## Multi-Attribute Funcs
##################################################

def train_multi_att(model, train_loader, label_dims, optimizer, loss_fn, use_gpu=False, device='cpu'):

    train_loss_list = []

    model.train()
    if use_gpu:
        model.to(device)

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.to(device)
            feature_lens = feature_lens.to(device)
            labels = {label_dim: labels[label_dim].to(device) for label_dim in label_dims}

        optimizer.zero_grad()

        preds,_ = model(features, feature_lens)

        ## multi-attribute
        loss = 0
        for label_dim in label_dims:
            loss += loss_fn(preds[label_dim].squeeze(-1), labels[label_dim].squeeze(-1))
        loss /= len(labels)

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    train_loss = np.mean(train_loss_list)
    return train_loss

def train_multi_att_model(task, model, data_loader, label_dims, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, regularization=0.0, device='cpu'):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    best_val_loss = float('inf')
    best_val_scores_mean = -1
    best_val_scores = {label_dim: -1 for label_dim in label_dims}
    best_model_file = ''
    early_stop = 0

    metrics = {"best_val_loss": best_val_loss,
                 "best_val_scores_mean": best_val_scores_mean,
                 "best_val_scores": best_val_scores,
                 "epoch": -1,
                 "train_loss": float('inf'),
                 "val_loss": float('inf'),
                 "val_scores_mean": -1,
                 "val_scores": {label_dim:-1 for label_dim in label_dims},}
    
    for epoch in range(1, epochs + 1):
        metrics.update({"epoch": epoch})        ## update epoch
        print(f'Training for Epoch {epoch}...')

        train_loss = train_multi_att(model, train_loader, label_dims, optimizer, loss_fn, use_gpu, device=device)
        metrics.update({"train_loss": train_loss})  ## update train_loss
        val_loss, val_scores = evaluate_multi_att(task, model, val_loader, label_dims, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu, device=device)
        val_scores_mean = np.mean(list(val_scores.values()))
        metrics.update({"val_loss": val_loss, "val_scores_mean": val_scores_mean})      ## update val_loss, val_scores
        for label_dim in label_dims:
            metrics["val_scores"].update({label_dim: val_scores[label_dim]})

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_scores_mean:>7.4f}, {val_scores}')
        print('-' * 50)

        if val_scores_mean > best_val_scores_mean:
            early_stop = 0
            best_val_scores_mean = val_scores_mean
            best_val_loss = val_loss
            metrics.update({"best_val_loss": best_val_loss, "best_val_scores_mean": best_val_scores_mean})      ## update best_val_score, best_val_loss
            for label_dim in label_dims:
                best_val_scores[label_dim] = val_scores[label_dim]
                metrics["best_val_scores"].update({label_dim: best_val_scores[label_dim]})
            # best_model_file = save_model(model, model_path, identifier)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        best_model_file = save_model(model, model_path, identifier)
        wandb.log(metrics)      ## log metrics

    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_scores_mean:>7.4f}, {best_val_scores} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_scores, best_model_file


##################################################
## Feature Fusion Funcs
##################################################

def train_feature_fusion(model, train_loader, optimizer, loss_fn, use_gpu=False, device='cpu'):

    train_loss_list = []

    model.train()
    if use_gpu:
        model.to(device)

    for batch, batch_data in enumerate(train_loader, 1):
        features, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        preds,_ = model(features)

        loss = loss_fn(torch.flatten(preds), torch.flatten(labels))

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    train_loss = np.mean(train_loss_list)
    return train_loss

def train_feature_fusion_model(task, model, data_loader, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, regularization=0.0, device='cpu'):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    metrics = {"best_val_loss": best_val_loss,
                 "best_val_score": best_val_score,
                 "epoch": -1,
                 "train_loss": float('inf'),
                 "val_loss": float('inf'),
                 "val_score": -1,}
    
    for epoch in range(1, epochs + 1):
        metrics.update({"epoch": epoch})        ## update epoch
        print(f'Training for Epoch {epoch}...')

        train_loss = train_feature_fusion(model, train_loader, optimizer, loss_fn, use_gpu, device=device)
        metrics.update({"train_loss": train_loss})  ## update train_loss
        val_loss, val_score = evaluate_feature_fusion(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu, device=device)
        metrics.update({"val_loss": val_loss, "val_score": val_score})      ## update val_loss, val_score

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            metrics.update({"best_val_score": best_val_score, "best_val_loss": best_val_loss})      ## update best_val_score, best_val_loss
            best_model_file = save_model(model, model_path, identifier)

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        wandb.log(metrics)      ## log metrics

    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_score, best_model_file
