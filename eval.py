import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from imblearn.core.utils import get_net_builder, get_dataset
import copy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.algorithms.supervised.utils import MARC_Net, Crt_Net, Lws_Net, DisAlign_Net

def shot_acc(training_labels, preds, labels, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='openai_clip_vit_large_patch14')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='/home/yzh/ILSVRC/')
    parser.add_argument('--dataset', type=str, default='imagenet_lt')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--decoder_depth', type=int, default=3, help='')
    parser.add_argument('--decoder_mlp_ratio', type=int, default=0.5, help='')
    parser.add_argument('--decoder_num_heads', type=int, default=4, help='')
    parser.add_argument('--output_dir', type=str, default='./logs', help='') 
    parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.')
 
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)

    if 'imagenet_lt' in args.load_path:
        args.dataset='imagenet_lt'
        args.num_classes = 1000
        args.data_dir = '/home/yzh/ILSVRC/'
    elif 'inaturalist' in args.load_path:
        args.dataset='inaturalist'
        args.num_classes = 8142
        args.data_dir = '/home/yzh/iNaturalist/'
    elif 'places' in args.load_path:
        args.dataset='places'
        args.num_classes = 365
        args.data_dir = '/home/yzh/Places/places365_standard/'
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes,decoder_depth=args.decoder_depth, decoder_mlp_ratio=args.decoder_mlp_ratio, decoder_num_heads=args.decoder_num_heads)
    if  'marc' in args.load_path: # https://arxiv.org/abs/2112.07225
        net = MARC_Net(args, net, args.num_classes)
    elif  'crt' in args.load_path: # https://arxiv.org/abs/1910.09217
        net = Crt_Net(args, net, args.num_classes)
    elif 'lws' in args.load_path: # https://arxiv.org/abs/1910.09217
        net = Lws_Net(args, net, args.num_classes)
    elif 'disalign' in args.load_path: # https://arxiv.org/abs/2103.16370
        net = DisAlign_Net(args, net, args.num_classes)
 
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    args.seed = 0
    dataset_dict = get_dataset(args, 'supervised',  args.dataset, args.num_classes, args.data_dir)
    if dataset_dict['test'] is not None:
        eval_dset = dataset_dict['test']
    else:
        eval_dset = dataset_dict['eval']
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
    training_labels = copy.deepcopy(dataset_dict['train_lb'].targets)
    training_labels = np.array(training_labels).astype(int)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in eval_loader:
            x = data['x_lb']
            y = data['y_lb']
            
            if isinstance(x, dict):
                x = {k: v.cuda() for k, v in x.items()}
            else:
                x = x.cuda()
            y = y.cuda()

            logits = net(x)['logits']
            
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    top1 = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    F1 = f1_score(y_true, y_pred, average='macro')
    many_shot_acc, median_shot_acc, few_shot_acc = shot_acc(training_labels ,y_pred, y_true)
   
    res_name = args.load_path.split('/')[-2]
    print(res_name)
    print("top1 accuracy:", top1)
    print("precision:", precision)
    print("recall:", recall)
    print("F1 score:", F1)
    print("many-shot accuracy:", many_shot_acc)
    print("median-shot accuracy:", median_shot_acc)
    print("few-shot accuracy:", few_shot_acc)

    output_file = os.path.join(args.output_dir, 'final_res.txt')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_file, 'a') as f:  # 将打开文件的模式设置为追加模式
        output_str = f"{res_name}\t{top1:.2f}\t{many_shot_acc:.2f}\t{median_shot_acc:.2f}\t{few_shot_acc:.2f}\t{precision:.2f}\t{recall:.2f}\t{F1:.2f}\n"
        f.write(output_str)
