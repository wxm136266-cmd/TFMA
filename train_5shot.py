import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import function.function as function
import time
from tqdm import tqdm
import os
from function.function import ContrastiveLoss, seed_func, convert_for_5shots, cal_accuracy_fewshot_ensemble_5shot
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.proposed_model import Ensemble_Net
import argparse
import torch.nn as nn
import numpy as np
import librosa
import cv2
import torch
from scipy.ndimage import gaussian_filter
from image_data_get import SAC_4way_image , create_set,create_set_test
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



seed_func()
parser = argparse.ArgumentParser(description='SAC Faults Project Configuration')
parser.add_argument('--dataset', default='SAC', help='Dataset (Only SAC)')
parser.add_argument('--training_samples_CWRU', type=int, default=30, help='Number of training samples for SAC')
parser.add_argument('--training_samples_PDB', type=int, default=195, help='Number of training samples for SAC')
parser.add_argument('--model_name', type=str, help='Model name')
parser.add_argument('--episode_num_train', type=int, default=100, help='Number of training episodes')
parser.add_argument('--episode_num_test', type=int, default=75, help='Number of testing episodes')
parser.add_argument('--way_num_CWRU', type=int, default=4, help='Number of classes for SAC')
parser.add_argument('--noise_DB', type=str, default=None, help='Noise database')
parser.add_argument('--way_num_PDB', type=int, default=13, help='Number of classes for SAC')
parser.add_argument('--spectrum', default=True, help='Use spectrum')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--path_weights', type=str, default='checkpoints/3channels/', help='Path to weights')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--loss1', default=ContrastiveLoss())
parser.add_argument('--loss2', default=nn.CrossEntropyLoss())
args = parser.parse_args()

print(args)

writer = SummaryWriter('./checkpoints/3channels/accloss/5shot-30samples_GC')

#---------------------------------------------------Load dataset-----------------------------------------------------------------------------------------:
if args.dataset == 'SAC':
    train_SAC,test_SAC = SAC_4way_image(way=4,example=50,spilt=30,shuffle=True)
    # train_SAC, test_SAC = SAC_4way_image_imbalance(way=4, example=50, spilt=50, shuffle=True)
    train_x_SAC ,train_y_SAC = create_set(train_SAC)
    # test_x_SAC,test_y_SAC = create_set_test(test_SAC)
    test_x_SAC, test_y_SAC = create_set(test_SAC)
    train_x_SAC = torch.from_numpy(np.array(train_x_SAC))
    train_y_SAC = torch.from_numpy(np.array(train_y_SAC))
    test_x_SAC = torch.from_numpy(np.array(test_x_SAC))
    test_y_SAC = torch.from_numpy(np.array(test_y_SAC))
    train_dataset_SAC = FewshotDataset(train_x_SAC ,train_y_SAC, episode_num=args.episode_num_train, way_num=args.way_num_CWRU, shot_num=5, query_num=2)
    train_dataloader_SAC = DataLoader(train_dataset_SAC, batch_size=args.batch_size, shuffle=True)
    test_dataset_SAC = FewshotDataset(test_x_SAC, test_y_SAC, episode_num=args.episode_num_test,way_num=args.way_num_CWRU, shot_num=5, query_num=2)
    test_dataloader_SAC = DataLoader(test_dataset_SAC, batch_size=args.batch_size, shuffle=False)

def train_and_test_model_ensemble(net,
                         train_dataloader,
                         test_loader,
                         training_samples,
                         num_epochs = args.num_epochs,
                         lr = args.lr,
                         loss1 = args.loss1,
                         loss2 = args.loss2,
                         path_weight = args.path_weights,
                         num_samples = args.training_samples_SAC):
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss1.to(device)
    loss2.to(device)
    full_loss = []
    full_acc = []
    pred_acc = 0
    global_count = 0
    test_global_count = 0
    cumulative_time = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        all_preds = []
        all_labels = []
        running_loss = 0
        num_batches = 0
        running_loss_1 = 0
        running_loss_2 = 0
        true_label = 0
        optimizer.zero_grad()
        print('='*50, 'Epoch:', epoch, '='*50)
        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as t:
            for query_images, query_targets, support_images, support_targets in t:
                global_count = global_count + 1

                q = query_images.permute(1, 0, 4, 2, 3).to(device)

                support_images = support_images.permute(0, 1, 4, 2, 3)

                s = convert_for_5shots(support_images, support_targets, device)

                targets = query_targets.to(device)
                targets = targets.permute(1, 0)
                for i in range(len(q)):
                    m_l, m_u, scores = net(q[i], s)
                    target = targets[i].long()
                    true_label += 1 if torch.argmax(scores) == target else 0
                    loss = loss1(m_l, target) + loss2(m_u, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    running_loss_1 += loss1(m_l, target).detach().item()
                    running_loss_2 += loss2(m_u, target).detach().item()
                    num_batches += 1

                    pred_label = torch.argmax(scores, dim=-1)
                    all_preds.append(pred_label.cpu().numpy())
                    all_labels.append(target.cpu().numpy())
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=running_loss / num_batches, loss_1 = running_loss_1 / num_batches, loss_2 = running_loss_2 / num_batches)
                writer.add_scalar('data/train_loss', float(running_loss / num_batches), global_count)
                writer.add_scalar('data/train_acc', float(true_label/num_batches), global_count)


            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)


            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')


            writer.add_scalar('data/train_precision', precision, global_count)
            writer.add_scalar('data/train_recall', recall, global_count)
            writer.add_scalar('data/train_f1', f1, global_count)
        elapsed_time = time.time() - start_time
        cumulative_time += elapsed_time
        cumulative_minutes = cumulative_time / 60
        print(f"Epoch {epoch}/{num_epochs} completed in {cumulative_minutes:.2f} minutes")

        scheduler.step()

        with torch.no_grad():
            test_global_count = test_global_count + 1
            total_loss = running_loss / num_batches
            full_loss.append(total_loss)
            print('------------Testing on the test set-------------')

            all_preds_test = []
            all_labels_test = []
            acc,test_loss,precision_test, recall_test, f1_test = cal_accuracy_fewshot_ensemble_5shot(test_loader, net, device,loss1, loss2)
            writer.add_scalar('data/val_loss', float(test_loss), test_global_count)
            writer.add_scalar('data/val_acc', float(acc), test_global_count)
            writer.add_scalar('data/val_precision', precision_test, test_global_count)
            writer.add_scalar('data/val_recall', recall_test, test_global_count)
            writer.add_scalar('data/val_f1', f1_test, test_global_count)
            full_acc.append(acc)
            print(f'Accuracy on the test set: {acc:.4f}')
            print(f'Accuracy_test: {acc:.4f}, Precision_test: {precision_test:.4f}, Recall_test: {recall_test:.4f}, F1-Score_test: {f1_test:.4f}')
            if acc > pred_acc:
                if epoch >= 2:
                    os.remove(path_weight + model_name)
                pred_acc = acc
                model_name = f'{args.model_name}_5shot_{acc:.4f}_{training_samples}samples_GC.pth'
                torch.save(net, path_weight + model_name)
                print(f'=> Save the best model with accuracy: {acc:.4f}')
        torch.cuda.empty_cache()
    model_end = f'{args.model_name}_5shot_{acc:.4f}_{training_samples}samples_end_GC.pth'
    torch.save(net, path_weight + model_end)
    return full_loss, full_acc


#----------------------------------------------------Training phase--------------------------------------------------#
if __name__ == "__main__":
    seed_func()
    net = Ensemble_Net()
    net = net.to(args.device)
    print('training.........................!!')
    if args.dataset == 'SAC':

        train_and_test_model_ensemble(net,
                                  train_dataloader=train_dataloader_SAC,
                                  test_loader=test_dataloader_SAC,
                                  training_samples=args.training_samples_CWRU,
                                  num_epochs=args.num_epochs,
                                  lr=args.lr,
                                  loss1=args.loss1,
                                  loss2=args.loss2,
                                  path_weight=args.path_weights,
                                  num_samples=args.training_samples_CWRU)

        print('end training...................!!')
    writer.close()
