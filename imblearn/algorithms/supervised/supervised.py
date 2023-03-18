# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from imblearn.core import AlgorithmBase
from imblearn.core.utils import ALGORITHMS, count_parameters
from imblearn.algorithms.supervised.utils import cbw_loss, focal_loss, balanced_softmax, grw_loss, lade_loss, ldam_loss, MARC_Net, Crt_Net, Lws_Net, DisAlign_Net
import torch

@ALGORITHMS.register('supervised')
class Supervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.loss_type = args.loss_type
        self.freeze_backbone = args.freeze_backbone
        self.extra_fc = args.extra_fc
        self.args = args

    def set_model(self):
        model = super().set_model()
        if 'stage1_path' in self.args:
            stage1_checkpoint = torch.load(self.args.stage1_path, map_location='cpu')
            model.load_state_dict(self.check_prefix_state_dict(stage1_checkpoint['model']))
            self.print_fn('Stage 1 Model loaded')
        if self.args.extra_fc == 'marc': # https://arxiv.org/abs/2112.07225
            model = MARC_Net(self.args, model, self.args.num_classes)
        if self.args.extra_fc == 'crt': # https://arxiv.org/abs/1910.09217
            model = Crt_Net(self.args, model, self.args.num_classes)
        if self.args.extra_fc == 'lws': # https://arxiv.org/abs/1910.09217
            model = Lws_Net(self.args, model, self.args.num_classes)
        if self.args.extra_fc == 'disalign': # https://arxiv.org/abs/2103.16370
            model = DisAlign_Net(self.args, model, self.args.num_classes)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes,decoder_depth=self.args.decoder_depth, decoder_mlp_ratio=self.args.decoder_mlp_ratio, decoder_num_heads=self.args.decoder_num_heads) 
        if self.args.extra_fc == 'marc':
            ema_model = MARC_Net(self.args, ema_model, self.args.num_classes)
        if self.args.extra_fc == 'crt':
            ema_model = Crt_Net(self.args, ema_model, self.args.num_classes)
        if self.args.extra_fc == 'lws':
            ema_model = Lws_Net(self.args, ema_model, self.args.num_classes)
        if self.args.extra_fc == 'disalign':
            ema_model = DisAlign_Net(self.args, ema_model, self.args.num_classes)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model


    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.freeze_backbone:
                if self.extra_fc != 'disalign':
                    with torch.no_grad(): 
                        outputs_x_lb = self.model(x_lb,only_feat=True)
                    logits_x_lb = self.model(outputs_x_lb,only_fc=True)
                else:
                    with torch.no_grad(): 
                        outputs_x_lb, feats_x_lb = self.model(x_lb,None,only_feat=True)
                    logits_x_lb = self.model(outputs_x_lb,feats_x_lb, only_fc=True)
 
            else:
                logits_x_lb = self.model(x_lb)['logits']
            if self.loss_type == 'softmax':
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            elif self.loss_type == 'cbw_loss': # class balanced reweighting, also check class balanced sampling in args.sapmle_type
                sup_loss = cbw_loss(logits_x_lb, y_lb, self.lb_freq, reduction='mean')
            elif self.loss_type == 'focal_loss': # https://arxiv.org/abs/1708.02002
                sup_loss = focal_loss(logits_x_lb, y_lb,gamma=2, reduction='mean')
            elif self.loss_type == 'balanced_softmax': # https://arxiv.org/abs/2007.10740
                sup_loss = balanced_softmax(logits_x_lb, y_lb,self.lb_freq, reduction='mean')
            elif self.loss_type == 'grw_loss': # https://arxiv.org/abs/2103.16370
                sup_loss = grw_loss(logits_x_lb, y_lb,self.lb_freq,exp_scale=1.2, reduction='mean')
            elif self.loss_type == 'lade_loss': # https://arxiv.org/abs/2012.00321
                sup_loss = lade_loss(logits_x_lb,y_lb,self.lb_freq,self.gpu, remine_lambda=0.1,estim_loss_weight = 0.1, reduction='mean')
            elif self.loss_type == 'ldam_loss': # https://arxiv.org/abs/1906.07413
                sup_loss = ldam_loss(logits_x_lb,y_lb,self.lb_cls_num,reduction='mean')
        
        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        for name, para in self.model.named_parameters():
            #if 'module.extra_mlp' in name or 'module.classifier' in name:
            if 'vision_model' in name:
                para.requires_grad = False
            else:
                para.requires_grad = True
            print(name, para.requires_grad)
        self.print_fn(f'Number of Trainable Params after freeze some parameters: {count_parameters(self.model)}')
        self.model.train()
        self.call_hook("before_run")
            
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")


