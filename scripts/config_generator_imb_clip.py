# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Create the .yaml for each experiment
"""
import os


def create_configuration(cfg, cfg_file):
    cfg['save_name'] = "{dataset}_{loss_type}_{sample_type}_{extra_fc}_{seed}".format(
        dataset=cfg['dataset'],
        loss_type=cfg['loss_type'],
        sample_type=cfg['sample_type'],
        extra_fc=cfg['extra_fc'],
        seed=cfg['seed'],
    )

    # resume
    cfg['resume'] = True
    cfg['load_path'] = '{}/{}/latest_model.pth'.format(cfg['save_dir'], cfg['save_name'])

    alg_file = cfg_file + cfg['algorithm'] + '/'
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    print(alg_file + cfg['save_name'] + '.yaml')
    with open(alg_file + cfg['save_name'] + '.yaml', 'w', encoding='utf-8') as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ': ' + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write('\n')


def create_classific_config(alg, seed,
                            dataset, data_dir, net, num_classes, 
                            port, weight_decay):
    cfg = {}
    cfg['algorithm'] = 'supervised'
    cfg['loss_type'] = alg['loss_type']
    cfg['sample_type'] = alg['sample_type']
    cfg['extra_fc'] = alg['extra_fc']
    
    
    # save config
    cfg['save_dir'] = './saved_models/imb_clip'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = True

    # algorithm config
    cfg['epoch'] = 100
    cfg['num_train_iter'] = 2 ** 20
    cfg['num_eval_iter'] = 2048
    cfg['num_log_iter'] = 256
    cfg['batch_size'] = 64
    cfg['eval_batch_size'] = 256
    # cfg['img']
    cfg['ema_m'] = 0.999
    cfg['crop_ratio'] = 0.875

    # optim config
    cfg['optim'] = 'SGD'
    cfg['lr'] = 0.03
    cfg['momentum'] = 0.9
    cfg['weight_decay'] = weight_decay
    cfg['layer_decay'] = 1.0
    cfg['amp'] = False
    cfg['clip'] = 0.0
    cfg['use_cat'] = True

    # net config
    cfg['net'] = net
    cfg['net_from_name'] = False

    # data config
    cfg['data_dir'] = data_dir
    cfg['dataset'] = dataset
    cfg['train_sampler'] = 'RandomSampler'
    cfg['num_classes'] = num_classes
    cfg['num_workers'] = 1

    # basic config
    cfg['seed'] = seed

    # distributed config
    cfg['world_size'] = 1
    cfg['rank'] = 0
    cfg['multiprocessing_distributed'] = True
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = None

    # other config
    cfg['overwrite'] = True
    cfg['amp'] = False
    cfg['use_wandb'] = False

    return cfg



# prepare the configuration for baseline model, use_penalty == False
def exp_imb_clip():
    config_file = r'./config/imb_clip/'
    save_path = r'./saved_models/imb_clip'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = [('imagenet_lt', '/mnt/sda/public/ILSVRC/'), ('places', '/mnt/sda/public/Places365_256/places365_standard/'), ('inaturalist', '/mnt/sda/public/iNaturalist18/')]


    # algs = ['flexmatch', 'fixmatch', 'uda', 'pseudolabel', 'fullysupervised', 'supervised', 'remixmatch', 'mixmatch', 'meanteacher',
    #         'pimodel', 'vat', 'dash', 'crmatch', 'comatch', 'simmatch', 'adamatch', 'freematch', 'softmatch']
    
    imb_algs = [
        {'loss_type': 'ldam_loss', 'sample_type': 'cbs', 'extra_fc': 'marc'},
        {'loss_type': 'ldam_loss', 'sample_type': 'cbs', 'extra_fc': 'marc'},
        {'loss_type': 'ldam_loss', 'sample_type': 'cbs', 'extra_fc': 'marc'},
    ]

    nets = [
        'openai_clip_vit_large_patch14'
    ]

    seeds = [0]

    dist_port = range(10001, 11120, 1)
    count = 0
    for net in nets:
        for alg in imb_algs:
            for dataset, data_dir in datasets:
                for seed in seeds:
                    # change the configuration of each dataset
                    if dataset == 'imagenet_lt':
                        # net = 'WideResNet'
                        num_classes = 1000
                        weight_decay = 5e-4

                    elif dataset == 'places':
                        # net = 'WideResNet'
                        num_classes = 365
                        weight_decay = 5e-4

                    elif dataset == 'inaturalist':
                        # net = 'WideResNet'
                        num_classes = 8142
                        weight_decay = 5e-4

                    port = dist_port[count]
                    # prepare the configuration file
                    cfg = create_classific_config(alg, seed, dataset, data_dir, net, num_classes, port, weight_decay)
                    count += 1
                    create_configuration(cfg, config_file)



if __name__ == '__main__':
    if not os.path.exists('./config/imb_clip/'):
        os.mkdir('./config/imb_clip/')

    exp_imb_clip()
