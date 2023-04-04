# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Create the .yaml for each experiment
"""
import os

commands = []

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
    
    run_script = f"python3 train.py --c {alg_file + cfg['save_name']}.yaml > {os.path.join('./logs', cfg['save_name'])}.log 2>&1\n"
    commands.append(run_script)
    print(run_script)

def create_classific_config(alg, seed,
                            dataset, data_dir, net, num_classes, 
                            port, weight_decay):
    cfg = {}
    cfg['algorithm'] = 'supervised'
    cfg['loss_type'] = alg['loss_type']
    cfg['sample_type'] = alg['sample_type']
    cfg['extra_fc'] = alg['extra_fc']
    if cfg['extra_fc'] is not None:
        cfg['freeze_backbone'] = True 
        cfg['stage1_path'] = "./saved_models/{dataset}_{loss_type}_{sample_type}_{extra_fc}_{seed}/model_best.pth".format(dataset=dataset,loss_type='softmax',sample_type=None,extra_fc=None,seed=seed)
    else:
        cfg['freeze_backbone'] = False
    # save config
    cfg['save_dir'] = './saved_models'
    cfg['save_name'] = None
    cfg['resume'] = False
    cfg['load_path'] = None
    cfg['overwrite'] = True
    cfg['use_tensorboard'] = True
    cfg['use_wandb'] = True

    # algorithm config
    cfg['epoch'] = 32
    cfg['num_warmup_iter'] = 512
    cfg['num_train_iter'] = 8192
    cfg['num_eval_iter'] = 256
    cfg['num_log_iter'] = 4
    cfg['batch_size'] = 256
    cfg['eval_batch_size'] = 256
    # cfg['img']
    cfg['ema_m'] = 0.0

    # optim config
    cfg['optim'] = 'SGD'
    if dataset == 'imagenet_lt':
        cfg['lr'] = 0.03
    elif dataset == 'places':
        cfg['lr'] = 0.03
    elif dataset == 'inaturalist':
        cfg['lr'] = 0.03


    cfg['momentum'] = 0.9
    cfg['weight_decay'] = weight_decay
    cfg['amp'] = False
    cfg['clip'] = 1.0

    cfg['net'] = net
    cfg['net_from_name'] = False

    # data config
    cfg['data_dir'] = data_dir
    cfg['dataset'] = dataset
    if cfg['sample_type'] == 'cbs':
        cfg['train_sampler'] = 'WeightedRandomSampler'
    else:
        cfg['train_sampler'] = 'RandomSampler'
    cfg['num_classes'] = num_classes
    cfg['num_workers'] = 12

    # basic config
    cfg['seed'] = seed

    # distributed config
    cfg['world_size'] = 1
    cfg['rank'] = 0
    cfg['multiprocessing_distributed'] = True
    cfg['dist_url'] = 'tcp://127.0.0.1:' + str(port)
    cfg['dist_backend'] = 'nccl'
    cfg['gpu'] = 0

    # other config
    cfg['overwrite'] = True
    cfg['amp'] = False
    cfg['use_wandb'] = True

    cfg['decoder_depth'] = 3
    cfg['decoder_mlp_ratio'] = 0.5
    cfg['decoder_num_heads'] = 4

    return cfg



# prepare the configuration for baseline model, use_penalty == False
def exp_imb_clip(config_file,imb_algs):
    #config_file = r'./config/imb_clip/'
    save_path = r'./saved_models'

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    datasets = [('imagenet_lt', '/home/yzh/ILSVRC/'), ('places', '/home/yzh/Places/places365_standard/'), ('inaturalist', '/home/yzh/iNaturalist/')]

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

    stage1_config_path = r'./config/imb_clip_stage1_algs/'
    stage2_config_path = r'./config/imb_clip_stage2_algs/'

    if not os.path.exists(stage1_config_path):
        os.mkdir(stage1_config_path)
    else:
        import shutil
        shutil.rmtree(stage1_config_path)
        os.mkdir(stage1_config_path)
    
    if not os.path.exists(stage2_config_path):
        os.mkdir(stage2_config_path)
    else:
        import shutil
        shutil.rmtree(stage2_config_path)
        os.mkdir(stage2_config_path)


    stage1_imb_algs = [
        {'loss_type': 'softmax', 'sample_type': None, 'extra_fc': None},
        {'loss_type': 'softmax', 'sample_type': 'cbs', 'extra_fc': None},
        {'loss_type': 'cbw_loss', 'sample_type': None, 'extra_fc': None},
        {'loss_type': 'focal_loss', 'sample_type': None, 'extra_fc': None},
        {'loss_type': 'balanced_softmax', 'sample_type': None, 'extra_fc': None},
        {'loss_type': 'grw_loss', 'sample_type': None, 'extra_fc': None},
        {'loss_type': 'lade_loss', 'sample_type': None, 'extra_fc': None},
        {'loss_type': 'ldam_loss', 'sample_type': None, 'extra_fc': None},
    ]
    exp_imb_clip(stage1_config_path,stage1_imb_algs)
    
    stage2_imb_algs = [
        {'loss_type': 'softmax', 'sample_type': 'cbs', 'extra_fc': 'crt'},
        {'loss_type': 'softmax', 'sample_type': 'cbs', 'extra_fc': 'lws'},
        {'loss_type': 'grw_loss', 'sample_type': None, 'extra_fc': 'disalign'},
        {'loss_type': 'grw_loss', 'sample_type': None, 'extra_fc': 'marc'},
    ]
    exp_imb_clip(stage2_config_path,stage2_imb_algs)

    with open('all_commands.txt', 'w') as f:
        f.writelines(commands)
