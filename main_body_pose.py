"""
    IMPORTING LIBS
"""

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

# from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.humanpose_graph_forecasting.load_net import gnn_model 
from data.data import LoadData 




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device










"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params, params, config):
    model = gnn_model(MODEL_NAME, net_params, params, config)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, gpus, config):

    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset['train'].name
    
    trainset, valset = dataset['train'], dataset['val']
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    # log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    # writer = SummaryWriter(log_dir=log_dir)
    # timeObj = time.localtime(time.time())
    # timeStr = '%d-%d-%d-%d-%d-%d' % (
    # timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)
    log_writer = open(os.path.join(root_log_dir, "log.txt"), mode="w", encoding="utf-8")

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Samples: ", len(trainset))
    print("Val Samples: ", len(valset))

    model = gnn_model(MODEL_NAME, net_params, params, config)

    if len(gpus) > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
    
    # import train and evaluate functions
    from train.train_human_pose import train_epoch, evaluate_network, evaluate_network_bayesian

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, epoch_train_div, epoch_train_ade, epoch_train_fde, optimizer = train_epoch(model, optimizer, device, train_loader, val_loader, epoch, params, MODEL_NAME)
                
                if MODEL_NAME == 'BayesianSpatioTemporalGraphTransformer':
                    epoch_val_loss, epoch_val_mae, epoch_val_div, epoch_val_ade, epoch_val_fde, epoch_val_mue = evaluate_network_bayesian(model, device, val_loader, epoch)
                else:
                    epoch_val_loss, epoch_val_mae, epoch_val_div, epoch_val_ade, epoch_val_fde = evaluate_network(model, device, val_loader, epoch)
                    epoch_val_mue = 0

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)

                log_string = 'epoch:'+str(epoch) 
                log_string += 'train_loss:'+str(epoch_train_loss) 
                log_string += 'train_mae:'+str(epoch_train_mae)
                log_string += 'val_mae:'+str(epoch_val_mae)

                log_string += 'train_div:'+str(epoch_train_div)
                log_string += 'val_div:'+str(epoch_val_div)

                log_string += 'train_ade:'+str(epoch_train_ade)
                log_string += 'val_ade:'+str(epoch_val_ade)

                log_string += 'train_fde:'+str(epoch_train_fde)
                log_string += 'val_fde:'+str(epoch_val_fde)

                log_string += 'val_mue:'+str(epoch_val_mue)
                
                log_string += 'lr:'+str(optimizer.param_groups[0]['lr'])

                log_writer.write(log_string)
                # writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                # writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                # writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                # writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                # writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                        
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae)


                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_mae = evaluate_network(model, device, val_loader, epoch)
    _, train_mae = evaluate_network(model, device, train_loader, epoch)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        




def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--dataset_dir', help="Please give a value for dataset dir")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--max_time', help="Please give a value for max_time")

    parser.add_argument('--in_channels', help="Please give a value for in_channels")
    parser.add_argument('--out_channels', help="Please give a value for out_channels")
    parser.add_argument('--dv_factor', help="Please give a value for dv_factor")
    parser.add_argument('--dk_factor', help="Please give a value for dk_factor")
    parser.add_argument('--Nh', help="Please give a value for Nh")
    parser.add_argument('--n', help="Please give a value for n")
    parser.add_argument('--relative', help="Please give a value for relative")
    parser.add_argument('--only_temporal_attention', help="Please give a value for only_temporal_attention")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--kernel_size_temporal', help="Please give a value for kernel_size_temporal")
    parser.add_argument('--stride', help="Please give a value for stride")
    parser.add_argument('--weight_matrix', help="Please give a value for weight_matrix")
    parser.add_argument('--last', help="Please give a value for last")
    parser.add_argument('--layer', help="Please give a value for layer")
    parser.add_argument('--more_channels', help="Please give a value for more_channels")
    parser.add_argument('--drop_connect', help="Please give a value for drop_connect")
    parser.add_argument('--num_point_in', help="Please give a value for num_point_in")
    parser.add_argument('--num_point_out', help="Please give a value for num_point_out")

    parser.add_argument('--len_input', help="Please give a value for len_input")
    parser.add_argument('--len_output', help="Please give a value for len_output")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("Using gpus: ", args.gpu_id)

    with open(args.config) as f:
        config = json.load(f)
            
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = args.gpu_id
        config['gpu']['use'] = True
        gpus = args.gpu_id.split(",")
    

    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    if args.dataset_dir is not None:
        DATASET_DIR = args.dataset_dir
    else:
        DATASET_DIR = config['dataset_dir']

    if args.len_input is not None:
        config['len_input'] = int(args.len_input)
    if args.len_output is not None:
        config['len_output'] = int(args.len_output)

    dataset = LoadData(DATASET_NAME, DATASET_DIR, config)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.in_channels is not None:
        net_params['in_channels'] = int(args.in_channels)
    if args.out_channels is not None:
        net_params['out_channels'] = int(args.out_channels)
    if args.dv_factor is not None:
        net_params['dv_factor'] = float(args.dv_factor)   
    if args.dk_factor is not None:
        net_params['dk_factor'] = float(args.dk_factor)
    if args.Nh is not None:
        net_params['Nh'] = int(args.Nh)
    if args.n is not None:
        net_params['n'] = int(args.n)
    if args.relative is not None:
        net_params['relative'] = True if args.relative=='True' else False
    if args.only_temporal_attention is not None:
        net_params['only_temporal_attention'] = True if args.only_temporal_attention=='True' else False
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.kernel_size_temporal is not None:
        net_params['kernel_size_temporal'] = int(args.kernel_size_temporal)
    if args.stride is not None:
        net_params['stride'] = int(args.stride)
    if args.weight_matrix is not None:
        net_params['weight_matrix'] = int(args.weight_matrix)
    if args.last is not None:
        net_params['last'] = True if args.last=='True' else False
    if args.layer is not None:
        net_params['layer'] = int(args.layer)
    if args.more_channels is not None:
        net_params['more_channels'] = True if args.more_channels=='True' else False
    if args.num_point_in is not None:
        net_params['num_point_in'] = int(args.num_point_in)
    if args.num_point_out is not None:
        net_params['num_point_out'] = int(args.num_point_out)
    if not os.path.exists(out_dir): 
        os.mkdir(out_dir)
    if not os.path.exists(out_dir + 'logs/'):
        os.mkdir(out_dir + 'logs/')
    if not os.path.exists(out_dir + 'checkpoints/'):
        os.mkdir(out_dir + 'checkpoints/')
    if not os.path.exists(out_dir + 'results/'):
        os.mkdir(out_dir + 'results/')
    if not os.path.exists(out_dir + 'configs/'):
        os.mkdir(out_dir + 'configs/')

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, params, config)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs, gpus, config)

    
    
    
    
    
    
    
main()    
















