# tensorboard --logdir=/logs.1H/NMR_1H
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
import json
import torch
import time
import time
import datetime
from tqdm import tqdm

os.environ["DGLBACKEND"] = "pytorch"
from typing import Dict

from architecture import NormUncertainLoss, ModelLoader
from data_preprocess import FeaturesConstansts
from utils import get_model_hyperparameters, get_test_loss, to_tensorboard, get_optimizer, get_scheduler, \
    Masks, DocumentationConstants, scale_labels


def set_documentation_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, DocumentationConstants.INFO_DIR.value))
        os.makedirs(os.path.join(path, DocumentationConstants.CHECKPOINTS_DIR.value))


def document_dict_as_json(dict_to_document: Dict, write_mode: str = 'w', json_path: str = None) -> None:
    if json_path:
        with open(json_path, write_mode) as file:
            json.dump(dict_to_document, file)


def document_dict_as_txt(dict_to_document: Dict, write_mode: str = 'w', txt_path: str = None) -> None:
    if txt_path:
        with open(txt_path, write_mode) as file:
            for key, value in dict_to_document.items():
                file.write(f'{key}: {value}\n')


def document_dict(dict_to_document: Dict, json_path: str = None, txt_path: str = None):
    document_dict_as_json(dict_to_document, 'w', json_path)
    document_dict_as_txt(dict_to_document, 'w', txt_path)


def document_dict_of_dict(dict_of_dict_to_document: Dict, json_path: str = None, txt_path: str = None):
    document_dict_as_json(dict_of_dict_to_document, 'w', json_path)
    if txt_path:
        for index, (key, value) in enumerate(dict_of_dict_to_document.items()):
            if index == 0:
                write_mode = 'w'
            with open(txt_path, write_mode) as file:
                file.write(f'{key}:\n')
            write_mode = 'a'
            document_dict_as_txt(value, write_mode, txt_path)


def document_object_as_pkl(object_to_document, pkl_path: str = None):
    import pickle
    if pkl_path:
        with open(pkl_path, 'wb') as file:
            pickle.dump(object_to_document, file)


def document_all_info(net_parameters, save_dir, features_meta_data):
    set_documentation_directory(save_dir)
    time.sleep(1)
    document_dict(net_parameters, os.path.join(save_dir, DocumentationConstants.INFO_JSON_PATH.value),
                  os.path.join(save_dir, DocumentationConstants.INFO_TXT_PATH.value))
    document_dict_of_dict(features_meta_data,
                          os.path.join(save_dir, DocumentationConstants.INFO_FEATURES_JSON_PATH.value),
                          os.path.join(save_dir, DocumentationConstants.INFO_FEATURES_TXT_PATH.value))
    document_object_as_pkl(FeaturesConstansts,
                           os.path.join(save_dir, DocumentationConstants.INFO_FEATURES_PICKLE_PATH.value))


def set_groundworks_for_training(net_parameters: Dict):
    nmr_type = net_parameters.get('nmr_type')
    save_dir = DocumentationConstants.MAIN_DIR.value.format(nmr_type,
                                                            datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    device = torch.device(net_parameters.get('device'))
    use_std = net_parameters.get('use_std')
    use_scheduler = net_parameters.get('scheduler').get('use_scheduler')
    features_set = net_parameters.get('features_set')

    model_loader = ModelLoader(log_dir=save_dir, net_parameters=net_parameters)
    model, train_dataloader, test_dataloader, train_dataset = model_loader.load_model()
    features_meta_data = model_loader.features_meta_data
    print('\033[94m' + 'Initializing NMR GatedGCN' + '\033[0m')
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: {}, Features set: {}".format(total_params, features_set))
    # time.sleep(1)
    if use_std:
        print("Using standard deviation in the loss function")
        criterion = NormUncertainLoss()
    else:
        print("Not using standard deviation in the loss function")
        criterion = nn.MSELoss()
    optimizer = get_optimizer(model, net_parameters)
    scheduler = get_scheduler(optimizer, net_parameters)
    # scheduler = WarmupScheduler(net_parameters['learning_rate']['start'], net_parameters['learning_rate']['max'],net_parameters['learning_rate']['epochs'])
    if use_scheduler:  # 'scheduler':
        print("Using scheduler {} with the kwargs {}".format(net_parameters.get('scheduler').get('type'),
                                                             net_parameters.get('scheduler').get('kwargs')))
    else:
        print("Not using scheduler")
    metrics = {'train_loss': torch.nan, 'test_loss': torch.nan, 'train_abs_error': torch.nan,
               'test_abs_error': torch.nan, 'train_max_error': torch.nan, 'test_max_error': torch.nan}
    if use_std:
        metrics['train_mean_std'] = torch.nan
        metrics['train_max_std'] = torch.nan
        metrics['train_min_std'] = torch.nan
        metrics['test_std_error'] = torch.nan
    document_all_info(net_parameters, save_dir, features_meta_data)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, DocumentationConstants.TENSORBOARD_DIR.value))
    print('\033[94m' + f'Training NMR GatedGCN on {device}' + '\033[0m')
    return model, train_dataloader, test_dataloader, device, use_std, use_scheduler, \
        writer, criterion, optimizer, scheduler, metrics, save_dir


def set_groundworks_for_epoch(dl_train, epoch, train_loss, test_loss, epoch_lr):
    pbar = tqdm(dl_train, total=len(dl_train), desc='Epoch {}'.format(epoch + 1))
    running_loss = 0.0
    running_abs_error = 0.0
    all_max_errors = []
    pbar.set_postfix({'Train Loss': train_loss, 'Test Loss': test_loss, 'LR': epoch_lr})
    return pbar, running_loss, running_abs_error, all_max_errors


def train_model(net_parameters):
    MIN_LABEL = net_parameters.get('MIN_LABEL')
    MAX_LABEL = net_parameters.get('MAX_LABEL')
    model, dl_train, dl_test, device, use_std, use_scheduler, writer, criterion, optimizer, scheduler, metrics, save_dir = \
        set_groundworks_for_training(net_parameters)
    time.sleep(2)
    start_time = time.time()
    # Training Loop
    for epoch in range(net_parameters.get('num_epochs')):
        lr = optimizer.param_groups[0]["lr"]
        pbar, running_loss, running_abs_error, all_max_errors = \
            set_groundworks_for_epoch(dl_train, epoch, metrics.get('train_loss'), metrics.get('test_loss'), lr)

        for i, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(pbar):
            ten_percent = int(len(dl_train) / 10)
            # batch to device
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['features'].to(device)
            batch_e = batch_graphs.edata['features'].to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = torch.cat(batch_labels, dim=0).to(device)
            batch_labels = scale_labels(batch_labels, MIN_LABEL, MAX_LABEL)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            mu, std = model(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            # add mask to labels
            labels_flat = batch_labels.reshape(-1, 1)
            mask = (labels_flat != Masks.MISSING_LABEL.value)

            # compute loss
            if use_std:
                loss = criterion(mu[mask], labels_flat[mask], std[mask])
            else:
                loss = criterion(labels_flat[mask], mu[mask])
            loss.backward()
            # clip_value = 1.0  # you can change this value
            # clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            # compute and log training metrics
            running_loss += loss.item()

            # rescale to compute errors
            abs_error = torch.abs(labels_flat[mask] - mu[mask])
            abs_error = abs_error * (MAX_LABEL - MIN_LABEL)
            running_abs_error += torch.mean(abs_error)
            all_max_errors.append(torch.max(abs_error))

            if i % ten_percent == 0:
                pbar.set_postfix(
                    {'Train Loss': running_loss / (i + 1), 'Test Loss': metrics.get('test_loss'), 'LR': lr}, )
        if use_scheduler:
            scheduler.step()

        metrics['avg_epoch_time'] = (time.time() - start_time) / (epoch + 1)

        metrics['test_loss'], metrics['test_abs_error'], \
            metrics['test_max_error'], metrics['test_mean_std'], \
            metrics['test_max_std'], metrics['test_min_std'] = get_test_loss(model,
                                                                             criterion, dl_test, device, MIN_LABEL,
                                                                             MAX_LABEL, use_std)
        metrics['train_loss'] = running_loss / len(dl_train)
        metrics['train_abs_error'] = running_abs_error / len(dl_train)
        metrics['train_max_error'] = torch.max(torch.stack(all_max_errors))
        if use_std:
            metrics['train_mean_std'] = torch.mean(std)
            metrics['train_max_std'] = torch.max(std)
            metrics['train_min_std'] = torch.min(std)

        if net_parameters.get('write_to_tensorboard'):
            to_tensorboard(writer, metrics, epoch, use_std)

        if (net_parameters['save_checkpoints'] and (epoch + 1) % net_parameters['checkpoint_every_k_epochs'] == 0):
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(save_dir, DocumentationConstants.CHECKPOINTS_PTH_PATH.value.format(str(epoch))))
            # f'checkpoints/checkpoint_epoch_{str(epoch)}.pth'))

    print('Finished Training')
    finish_time = time.time() - start_time
    print('total time: {}'.format(finish_time))
    # Close the TensorBoard writer
    writer.close()
    return model, dl_train, dl_test


class ModelRunner(ModelLoader):
    def __init__(self, log_dir: str = None, net_parameters: Dict = None) -> None:
        super(ModelRunner, self).__init__(log_dir, net_parameters)

    def train_model(self):
        self.model, self.train_dataloader, self.test_dataloader = train_model(self.net_parameters)


if __name__ == '__main__':
    net_parameters = get_model_hyperparameters()
    model_runner = ModelRunner(net_parameters=net_parameters)
    model_runner.train_model()
    # model, dl_train, dl_test = train_model(net_parameters)