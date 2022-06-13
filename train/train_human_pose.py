"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE, get_beta, compute_mue, compute_diversity, compute_ade, compute_fde

def model_loss(scores, targets):
    # loss = nn.MSELoss()(scores,targets)
    loss = nn.L1Loss()(scores, targets)
    return loss

def train_epoch(model, optimizer, device, data_loader, val_loader, epoch, params, MODEL_NAME):
    model.train()
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_kl_loss = 0

    epoch_train_mae = 0
    epoch_train_div = 0
    epoch_train_ade = 0
    epoch_train_fde = 0

    nb_data = 0
    gpu_mem = 0
    for iter, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()

        if MODEL_NAME == 'BayesianSpatioTemporalGraphTransformer':
            batch_predictions, kl = model.forward(batch_inputs)
            loss = model_loss(batch_predictions, batch_targets)
            beta = get_beta(iter, len(data_loader), params['beta_type'], epoch, params['epochs'])
            kl_loss = beta * kl.mean()
            # print("kl_loss:", kl_loss)
            epoch_kl_loss += kl_loss.item()
            loss += kl_loss
        else:
            batch_predictions = model.forward(batch_inputs)
            loss = model_loss(batch_predictions, batch_targets)
        # print("batch_predictions:", batch_predictions.shape)
        # print("batch_targets:", batch_targets.shape)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_mae_loss += loss.item()
        epoch_train_mae += MAE(batch_predictions, batch_targets)
        epoch_train_div += compute_diversity(batch_predictions)
        epoch_train_ade += compute_ade(batch_predictions, batch_targets)
        epoch_train_fde += compute_fde(batch_predictions, batch_targets)

        nb_data += batch_targets.size(0)

        # for debug
        # epoch_val_loss, epoch_val_mae, epoch_val_div, epoch_val_ade, epoch_val_fde, epoch_val_mue = evaluate_network_bayesian(model, device, val_loader, epoch)
        # print("val:", epoch_val_loss, epoch_val_mae, epoch_val_div, epoch_val_ade, epoch_val_fde, epoch_val_mue)

        if (iter + 1) % 100 == 0:
            print("epoch:", epoch, "iter:", iter, "total_loss:", epoch_loss/(iter + 1), "mae_loss:", epoch_mae_loss/(iter + 1), "kl_loss:", epoch_kl_loss/(iter + 1))
            print("train_mae:", epoch_train_mae/(iter + 1), "train_div:", epoch_train_div/(iter + 1), "train_ade:", epoch_train_ade/(iter + 1), "train_fde:", epoch_train_fde/(iter + 1))
            
            # validation
            if MODEL_NAME == 'BayesianSpatioTemporalGraphTransformer':
                epoch_val_loss, epoch_val_mae, epoch_val_div, epoch_val_ade, epoch_val_fde, epoch_val_mue = evaluate_network_bayesian(model, device, val_loader, epoch)
            else:
                epoch_val_loss, epoch_val_mae, epoch_val_div, epoch_val_ade, epoch_val_fde = evaluate_network(model, device, val_loader, epoch)
                epoch_val_mue = 0
            
            print("val_loss:", epoch_val_loss, "val_mae:", epoch_val_mae, "val_div:", epoch_val_div)
            print("val_ade:", epoch_val_ade, "val_fde:", epoch_val_fde, "val_mue:", epoch_val_mue)


    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_div /= (iter + 1) 
    epoch_train_ade /= (iter + 1)
    epoch_train_fde /= (iter + 1)

    return epoch_loss, epoch_train_mae, epoch_train_div, epoch_train_ade, epoch_train_fde, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_div = 0
    epoch_test_ade = 0
    epoch_test_fde = 0

    nb_data = 0
    with torch.no_grad():
        for iter, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
                
            batch_predictions = model.forward(batch_inputs)
            loss = model_loss(batch_predictions, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_predictions, batch_targets)
            epoch_test_div += compute_diversity(batch_predictions)
            epoch_test_ade += compute_ade(batch_predictions, batch_targets)
            epoch_test_fde += compute_fde(batch_predictions, batch_targets)

            nb_data += batch_targets.size(0)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_div /= (iter + 1)
        epoch_test_ade /= (iter + 1)
        epoch_test_fde /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, epoch_test_div, epoch_test_ade, epoch_test_fde

def evaluate_network_bayesian(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    epoch_test_div = 0
    epoch_test_ade = 0
    epoch_test_fde = 0
    epoch_test_mue = 0

    nb_data = 0
    with torch.no_grad():
        for iter, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
                
            batch_predictions, uncertainty = model(x=batch_inputs, training=False)
            # print("batch_predictions[0]:", batch_predictions[0].shape)
            # print("uncertainty:", uncertainty[0])

            loss = model_loss(batch_predictions[0], batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_predictions[0], batch_targets)
            epoch_test_div += compute_diversity(batch_predictions[0])
            epoch_test_ade += compute_ade(batch_predictions[0], batch_targets)
            epoch_test_fde += compute_fde(batch_predictions[0], batch_targets)
            epoch_test_mue += compute_mue(batch_predictions, batch_targets, uncertainty)

            nb_data += batch_targets.size(0)
            
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_div /= (iter + 1)
        epoch_test_ade /= (iter + 1)
        epoch_test_fde /= (iter + 1)
        epoch_test_mue /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, epoch_test_div, epoch_test_ade, epoch_test_fde, epoch_test_mue
