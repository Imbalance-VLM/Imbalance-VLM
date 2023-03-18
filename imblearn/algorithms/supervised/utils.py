import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def balanced_softmax(logits,targets, freq, reduction='mean'):
    freq = freq.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + freq.log()
    return F.cross_entropy(logits, targets, reduction=reduction)


def focal_loss(logits, targets, gamma=2,reduction='mean'):
    pred = F.softmax(logits, dim=-1)
    targets = F.one_hot(targets, num_classes=logits.shape[1])
    loss = -targets*((1-pred)**gamma)*torch.log(pred)
    loss = torch.sum(loss,dim=-1)
    loss = loss.mean()
    return loss

def cbw_loss(logits, targets, freq, reduction='mean'):
    weight = freq.sum()/(freq.shape[0]*freq)
    weight = weight.type(torch.float32)
    return F.cross_entropy(logits, targets, weight=weight, reduction=reduction)

def grw_loss(logits, targets, freq, exp_scale=1.2, reduction='mean'):
    num_classes = logits.shape[1]
    exp_reweight = 1 / (freq ** exp_scale)
    exp_reweight = exp_reweight / torch.sum(exp_reweight)
    exp_reweight = exp_reweight * num_classes
    exp_reweight = exp_reweight.type(torch.float32)
    return F.cross_entropy(logits,targets,weight=exp_reweight,reduction=reduction)

def mine_lower_bound(x_p, x_q, num_samples_per_cls):
    N = x_p.size(-1)
    first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
    second_term = torch.logsumexp(x_q, -1) - np.log(N)

    return first_term - second_term, first_term, second_term

def remine_lower_bound(x_p, x_q, num_samples_per_cls, remine_lambda):
    loss, first_term, second_term = mine_lower_bound(x_p, x_q, num_samples_per_cls)
    reg = (second_term ** 2) * remine_lambda
    return loss - reg, first_term, second_term

def lade_loss(logits,targets,freq,gpu,remine_lambda =0.1, estim_loss_weight = 0.1, reduction='mean'):
    cls_weight = freq
    freq = freq.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + freq.log()
    priorce_loss =  F.cross_entropy(logits, targets, reduction=reduction)
    
    num_classes = logits.shape[1]
    balanced_freq = torch.tensor(1. / num_classes).float().cuda(gpu)
    per_cls_pred_spread = logits.T * (targets == torch.arange(0, num_classes).view(-1, 1).type_as(targets))  # C x N
    pred_spread = (logits - torch.log(freq + 1e-9) + torch.log(balanced_freq + 1e-9)).T  # C x N
    num_samples_per_cls = torch.sum(targets == torch.arange(0, num_classes).view(-1, 1).type_as(targets), -1).float()  # C
    estim_loss, first_term, second_term = remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls, remine_lambda)
    estim_loss = -torch.sum(estim_loss * cls_weight)     
    
    return priorce_loss + estim_loss_weight * estim_loss



def ldam_loss(logits,targets,cls_num,max_m=0.5,s=30,reduction='mean'):
    m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num))
    m_list = m_list * (max_m / torch.max(m_list))
    index = torch.zeros_like(logits,dtype=torch.uint8)
    index.scatter_(1, targets.data.view(-1, 1), 1)   
    index_float = index.type(torch.cuda.DoubleTensor)
    batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
    batch_m = batch_m.view((-1, 1))
    logits_m = logits - batch_m
    output = torch.where(index, logits_m, logits)
    return F.cross_entropy(output, targets)




class Crt_Net(nn.Module):
    def __init__(self,args, base, num_classes):
        super(Crt_Net, self).__init__()
        self.args = args
        self.backbone = base
        self.num_classes = num_classes
        if 'wrn' in self.args.net or 'resnet' in self.args.net:
            self.backbone.classifier = nn.Linear(self.backbone.num_features, self.num_classes)
        elif 'vit' in self.args.net:
            self.backbone.head =  nn.Linear(self.backbone.num_features, self.num_classes)  
    def forward(self, x, only_feat=False, only_fc=False):
        if only_feat is True:
            return self.backbone(x,only_feat=True)
        if only_fc is True:
            return self.backbone(x,only_fc=True)
        else:
            return {'logits': self.backbone(x)['logits']}
class Lws_Net(nn.Module):
    def __init__(self,args, base, num_classes):
        super(Lws_Net, self).__init__()
        self.args = args
        self.backbone = base
        self.num_classes = num_classes
        self.scales = nn.Parameter(torch.ones(args.num_classes))
        
    def forward(self, x, only_feat=False, only_fc=False):
        if only_feat is True:
            return self.backbone(x,only_feat=True)
        if only_fc is True:
            return self.scales * self.backbone(x,only_fc=True)
        else:
            return {'logits': self.scales * self.backbone(x)['logits']}

class DisAlign_Net(nn.Module):
    def __init__(self,args, base, num_classes):
        super(DisAlign_Net, self).__init__()
        self.args = args
        self.backbone = base
        self.num_classes = num_classes
        self.logit_scale = nn.Parameter(torch.ones(1, args.num_classes))
        self.logit_bias = nn.Parameter(torch.zeros(1, args.num_classes))
        self.confidence_layer = nn.Linear(self.backbone.num_features, 1)
 
    def forward(self, x, feat, only_feat = False, only_fc = False):
        if only_feat is True:
            feats_x = self.backbone(x,only_feat=True)
            logits_x = self.backbone(feats_x,only_fc=True) 
            return logits_x,feats_x
        if only_fc is True:
            confidence = self.confidence_layer(feat).sigmoid()
            return (1 + confidence * self.logit_scale) * x + confidence * self.logit_bias
        else:
            feats_x = self.backbone(x,only_feat=True)
            logits_x = self.backbone(feats_x,only_fc=True)
            confidence = self.confidence_layer(feats_x).sigmoid()
            return {'logits': (1 + confidence * self.logit_scale) * logits_x + confidence * self.logit_bias}

 


class MARC_Net(nn.Module):
    def __init__(self,args, base, num_classes):
        super(MARC_Net, self).__init__()
        self.args = args
        self.backbone = base
        self.num_classes = num_classes
        self.a = torch.nn.Parameter(torch.ones(1, self.num_classes))
        self.b = torch.nn.Parameter(torch.zeros(1, self.num_classes))
        if 'wrn' in self.args.net or 'resnet' in self.args.net:
            self.w_norm = torch.norm(self.backbone.classifier.weight.data,dim=1).cuda(self.args.gpu)
        elif 'clip' in self.args.net:
            self.w_norm = torch.norm(self.backbone.classifier.weight.data,dim=1).cuda(self.args.gpu)
        elif 'vit' in self.args.net:
            self.w_norm = torch.norm(self.backbone.head.weight.data,dim=1).cuda(self.args.gpu)
 
    def forward(self, x, only_feat = False, only_fc = False):
        if only_feat is True:
            return self.backbone(x)
        if only_fc is True:
            return self.a * x['logits'] + self.b * self.w_norm
        else:
            logits = self.backbone(x)['logits']
            return {'logits': self.a *logits + self.b * self.w_norm }
