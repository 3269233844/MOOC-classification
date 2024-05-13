import numpy as np
import torch
import math
import torch.nn as nn
from transformers import get_scheduler
from transformers import BertModel
from sklearn import metrics


# 下游网络权重初始化，默认xavier
def init_network(model):
    for child in model.children():
        for p in child.parameters():
            if type(child) != BertModel:
                if p.requires_grad:
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)


def train_model(config, model, train_dataloader, dev_dataloader, test_dataloader):
    model.train()
    total_epoch = 0  # 记录进行到多少轮
    dev_best_loss = float('inf')
    dev_best_acc = 0
    dev_best_f1 = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    train_loss_total = 0

    # 定义优化器
    if config.use_adam_scheduler:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        lr_scheduler = get_scheduler(name='linear',
                                     optimizer=optimizer,
                                     num_warmup_steps=int(config.epoch * len(train_dataloader) * 0.05),
                                     num_training_steps=int(config.epoch * len(train_dataloader)))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print("----------开始训练----------")
    model.train()
    for e in range(config.epoch):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_loss, train_pre = model.forward(batch)
            train_loss.backward()
            optimizer.step()
            if config.use_adam_scheduler:
                lr_scheduler.step()

            train_loss_total = train_loss_total + train_loss

        train_loss_epoch = train_loss_total / len(train_dataloader)
        dev_acc, dev_f1, dev_loss = to_test_model(config, model, dev_dataloader, is_test=False)  # 验证集输出

        if config.acc_save:
            if dev_best_acc < dev_acc:
                dev_best_acc = dev_acc
                torch.save(model.state_dict(), config.model_save_path)  # 保存模型
                improve = '模型已保存'
                last_improve = total_epoch
            else:
                improve = ''

        elif config.f1_save:
            if dev_best_f1 < dev_f1:
                dev_best_f1 = dev_f1
                torch.save(model.state_dict(), config.model_save_path)  # 保存模型
                improve = '模型已保存'
                last_improve = total_epoch
            else:
                improve = ''

        else:
            if dev_best_loss > dev_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(), config.model_save_path)  # 保存模型
                improve = '模型已保存'
                last_improve = total_epoch
            else:
                improve = ''
        total_epoch += 1
        msg = "total_epoch:{}/{}: train_loss_epoch:{} dev_loss:{} dev_f1:{} dev_acc:{} {}"
        print(msg.format(total_epoch, config.epoch, train_loss_epoch.item(), dev_loss, dev_f1, dev_acc, improve))
        model.train()
        train_loss_total = 0
        if total_epoch - last_improve > config.patience:
            print("-----验证集指标超过patience没改善，结束训练-----")
            break
    model.load_state_dict(torch.load(config.model_save_path))
    to_test_model(config, model, test_dataloader, is_test=True)


def to_test_model(config, model, dataloader, is_test=True):
    if is_test:
        print("----------开始测试----------")
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            loss, pre = model.forward(batch)
            loss_total = loss_total + loss
            labels = batch['label'].squeeze(1).data.cpu().numpy()
            if config.use_bce_loss:
                predict = torch.round(torch.sigmoid(pre)).data.cpu().numpy()
            else:
                predict = torch.max(pre.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    if is_test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        print(report)
    else:
        acc = metrics.accuracy_score(labels_all, predict_all)
        f1 = metrics.f1_score(labels_all, predict_all, average='weighted')
        return acc, f1, loss_total / len(dataloader)


