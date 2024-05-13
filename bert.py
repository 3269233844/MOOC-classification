import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertSelfAttention, BertPooler
from capsule_layer import CapsuleLinear


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'bert_mpcg'
        self.model_save_path = "model_save/" + self.model_name + ".pth"
        self.model_pretrain_path = "./bert_pretrain"
        self.model_vocab_path = "./bert_pretrain"
        self.file_path = "data/stanfordMOOCForumPostsSet.csv"

        self.bert_param_requires_grad = True
        self.use_bce_loss = False  # 使用指定的损失函数
        self.use_adam_scheduler = True  # 使用指定优化器
        self.acc_save = True
        self.f1_save = False
        self.GroupA = True
        self.GroupB = False  # 均为False时使用GroupC
        self.re_data = False  # 进行数据增强
        self.collate_length_processing = False  # 使不同的batch_size文本长度可以不同

        self.bert_embedding_dim = 768
        self.max_len = 270  # 最大句子长度
        self.batch_size = 16  # batch_size大小
        self.epoch = 10  # 训练轮次
        self.patience = 2  # 超过patience数效果未提升，则提前训练结束，防止过拟合
        self.lr = 2e-5  # 学习率
        self.weight_decay = 0
        self.class_list = ['No urgent', 'Urgent']  # 类别名单
        self.class_num = 2  # 分类数
        self.dropout = 0.14

        self.cnns_channels = 768  # 串行卷积核数量(out_channels数)
        self.gate_size = 768 * 2
        self.agg_out_channels = 350  # 卷积核数量(out_channels数)
        self.filter_sizes = [2, 3, 4]  # 并行卷积核尺寸
        self.kernel_size = 768  # 并行卷积核尺寸
        self.out_channels = 256  # 并行卷积核数量(out_channels数)
        self.max_rnn_length = self.max_len - max(self.filter_sizes) + 1  # 输入rnn的长度
        self.hidden_size = len(self.filter_sizes) * self.out_channels  # rnn单元的维度
        self.rnn_hidden = 256  # rnn每层的单元数
        self.num_layers = 2  # rnn的层数
        self.classifier_size = 768 * 2 + self.rnn_hidden * 2  # 分类器输入大小
        self.test_size = 0.33  # 测试集占数据集的比例
        self.dev_size = 0.3  # 验证集占测试集的比例

        self.SEED = 2604  # 保证每次结果一样
        # self.seed_save_path = 'model_save/random_state_{}'.format(self.SEED)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_vocab_path)  # 使用指定的分词器，路径内包含词表

        if self.GroupA:
            self.model_save_path = "model_save/" + self.model_name + "_1groupA" + str(self.SEED) + ".pth"
        elif self.GroupB:
            self.model_save_path = "model_save/" + self.model_name + "_0groupB" + str(self.SEED) + ".pth"
        else:
            self.model_save_path = "model_save/" + self.model_name + "_groupC" + str(self.SEED) + ".pth"


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_pretrain_path, return_dict=False)
        for param in self.bert.parameters():
            param.requires_grad = config.bert_param_requires_grad

        self.cnns_first = nn.Conv2d(config.bert_embedding_dim, config.cnns_channels, kernel_size=(1, 1))
        self.cnns_end = nn.Conv2d(config.cnns_channels, config.bert_embedding_dim, kernel_size=(1, 1))

        # self.cnn = nn.Conv2d(1, 768, kernel_size=(1, 768))
        # self.rnn1 = nn.GRU(768, 256, 2, bidirectional=True, batch_first=True, dropout=config.dropout)

        self.bert_self_attention = BertSelfAttention(self.bert.config)
        self.bert_pooler = BertPooler(self.bert.config)
        self.gate = nn.Sequential(
            nn.Linear(config.gate_size, config.gate_size),
            nn.Sigmoid()
        )

        self.conv_agg = nn.Conv2d(1, config.agg_out_channels, (1, config.bert_embedding_dim))
        # self.pwc = nn.Conv2d(768, 768, kernel_size=(1, 1))
        self.convs = nn.ModuleList([nn.Conv2d(1, config.out_channels, (k, config.agg_out_channels)) for k in config.filter_sizes])  # 并行卷积
        self.padding1 = nn.ZeroPad2d((0, 0, 0, 1))  # 底部零填充
        self.rnn = nn.GRU(config.hidden_size, config.rnn_hidden, config.num_layers, bidirectional=True, batch_first=True, dropout=0.12)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(0.12)
        self.cat_linear = nn.Linear(config.bert_embedding_dim * 2, config.bert_embedding_dim)

        self.classifier3 = nn.Linear(config.classifier_size, config.class_num)
        if config.use_bce_loss:
            self.classifier = nn.Linear(config.classifier_size, 1)
            self.loss_fun = nn.BCEWithLogitsLoss()
        else:
            self.classifier = nn.Linear(config.classifier_size, config.class_num)
            self.loss_fun = nn.CrossEntropyLoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3).permute(0, 2, 1)
        # x = self.padding1(x)
        # x = self.padding1(x)
        # x = self.padding1(x)
        x = x[:, :self.max_rnn_length, :]
        return x

    def forward(self, batch):
        text_ids = batch['text_ids'].to(self.config.device)
        label = batch['label'].to(self.config.device)
        mask = batch['mask'].to(self.config.device)

        encoder_out, text_cls = self.bert(text_ids, attention_mask=mask)

        # 全局语义细化
        cnn_first_out = self.relu(self.cnns_first(encoder_out.unsqueeze(3).permute(0, 2, 1, 3)))  # (b,n,d) -> (b,n,d,1) -> (b,d,n,1) -> (b,c,n,1)
        cnn_end_out = self.relu(self.cnns_end(cnn_first_out)).permute(0, 2, 1, 3).squeeze(3)  # (b,c,n,1) -> (b,d,n,1) -> (b,n,d,1)
        dpc_out = self.dropout(cnn_end_out)
        cat_out = self.cat_linear(torch.cat([encoder_out, dpc_out], -1))
        bert_self_attention_out = self.dropout(self.bert_self_attention(cat_out)[0])

        # cnn_out = F.relu(self.cnn(bert_self_attention_out.unsqueeze(1))).squeeze(3)
        # cnn_pool = F.max_pool1d(cnn_out, cnn_out.size(2)).squeeze(2)

        # rnn1_out, _ = self.rnn1(bert_self_attention_out)
        # rnn1_out = self.dropout(rnn1_out[:, -1, :])

        # max_out = bert_self_attention_out.permute(0, 2, 1)
        # max_pool = F.max_pool1d(max_out, max_out.size(2)).squeeze(2)

        bert_pool = self.bert_pooler(bert_self_attention_out)
        average_pool = bert_self_attention_out.mean(dim=1)

        act_out = torch.cat([bert_pool, average_pool], -1)
        gate_out = self.gate(act_out)

        # 局部语义提取
        agg_out = self.relu(self.conv_agg(encoder_out.unsqueeze(1))).squeeze(3).permute(0, 2, 1)
        # agg_out = self.relu(self.pwc(encoder_out.unsqueeze(3).permute(0, 2, 1, 3))).squeeze(3).permute(0, 2, 1)
        cnn_out = torch.cat([self.conv_and_pool(agg_out.unsqueeze(1), conv) for conv in self.convs], -1)
        rnn_out, hidden = self.rnn(cnn_out)
        rnn_out = self.dropout(rnn_out[:, -1, :])
        # rnn_out = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # rnn_out = torch.tanh(rnn_out)
        # rnn_out = F.relu(rnn_out)

        # 分类向量
        out = torch.cat([gate_out, rnn_out], 1)
        pre = self.classifier(out).squeeze(1)
        loss = self.loss_fun(pre, label.squeeze(1))

        return loss, pre



# class TextModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.bert = BertModel.from_pretrained(config.model_pretrain_path, return_dict=False)
#         for param in self.bert.parameters():
#             param.requires_grad = config.bert_param_requires_grad
#
#         self.cnns_first = nn.Conv2d(config.bert_embedding_dim, config.cnns_channels, kernel_size=(1, 1))
#         self.cnns = nn.ModuleList([nn.Conv2d(config.cnns_channels, config.cnns_channels, kernel_size=(1, 1)) for _ in range(config.cnns_numbers)])  # 串行点卷积
#         self.cnns_end = nn.Conv2d(config.cnns_channels, config.bert_embedding_dim, kernel_size=(1, 1))
#
#         self.bert_self_attention = BertSelfAttention(self.bert.config)
#         self.bert_pooler = BertPooler(self.bert.config)
#         self.gate = nn.Sequential(
#             nn.Linear(config.gate_size, config.gate_size),
#             nn.Sigmoid()
#         )
#
#         self.conv_agg = nn.Conv2d(1, config.agg_out_channels, (1, config.bert_embedding_dim))
#         self.convs = nn.ModuleList([nn.Conv2d(1, config.out_channels, (k, config.agg_out_channels)) for k in config.filter_sizes])  # 并行卷积
#         self.rnn = nn.GRU(config.hidden_size, config.rnn_hidden, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(config.dropout)
#         self.cat_linear = nn.Linear(config.bert_embedding_dim * 2, config.bert_embedding_dim)
#
#         self.classifier3 = nn.Linear(config.classifier_size, config.class_num)
#         self.lsr = CrossEntropyLoss_LSR(config.device)
#         if config.use_bce_loss:
#             self.classifier = nn.Linear(config.classifier_size, 1)
#             self.loss_fun = nn.BCEWithLogitsLoss()
#         else:
#             self.classifier = nn.Linear(config.classifier_size, config.class_num)
#             self.loss_fun = nn.CrossEntropyLoss()
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3).permute(0, 2, 1)
#         x = x[:, :self.config.max_rnn_length, :]
#         return x
#
#     def forward(self, batch):
#         text_ids = batch['text_ids'].to(self.config.device)
#         label = batch['label'].to(self.config.device)
#         mask = batch['mask'].to(self.config.device)
#
#         encoder_out, text_cls = self.bert(text_ids, attention_mask=mask)
#
#         # serial cnn 串行卷积
#         cnn_first_out = self.relu(self.cnns_first(encoder_out.unsqueeze(3).permute(0, 2, 1, 3)))  # (b,n,d) -> (b,n,d,1) -> (b,d,n,1) -> (b,c,n,1)
#         if len(self.cnns):
#             cnns_out = cnn_first_out
#             for cnn in self.cnns:
#                 cnns_out = self.relu(cnn(cnns_out))
#             cnn_end_out = self.relu(self.cnns_end(cnns_out)).permute(0, 2, 1, 3).squeeze(3)  # (b,c,n,1) -> (b,d,n,1) -> (b,n,d,1)
#         else:
#             cnn_end_out = self.relu(self.cnns_end(cnn_first_out)).permute(0, 2, 1, 3).squeeze(3)
#         serial_cnn_out = self.dropout(cnn_end_out)
#
#         # 注意力， 池化， Gate融合
#         cat_out = self.cat_linear(torch.cat([encoder_out, serial_cnn_out], -1))
#         bert_self_attention_out = self.dropout(self.bert_self_attention(cat_out)[0])
#         bert_pool = self.bert_pooler(bert_self_attention_out)
#         average_pool = bert_self_attention_out.mean(dim=1)
#         gate_out = self.gate(torch.cat([bert_pool, average_pool], -1))
#
#         # 并行卷积
#         agg_out = F.relu(self.conv_agg(encoder_out.unsqueeze(1))).squeeze(3).permute(0, 2, 1)
#         cnn_out = torch.cat([self.conv_and_pool(agg_out.unsqueeze(1), conv) for conv in self.convs], -1)
#         rnn_out, _ = self.rnn(cnn_out)
#         rnn_out = self.dropout(rnn_out[:, -1, :])
#
#         out = torch.cat([gate_out, rnn_out], 1)
#         pre = self.classifier(out).squeeze(1)
#         loss = self.loss_fun(pre, label.squeeze(1))
#
#         return loss, pre


# class TextModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.bert = BertModel.from_pretrained(config.model_pretrain_path, return_dict=False)
#         for param in self.bert.parameters():
#             param.requires_grad = config.bert_param_requires_grad
#
#         self.cnns_first = nn.Conv2d(config.bert_embedding_dim, config.cnns_channels, kernel_size=(1, 1))
#         self.cnns = nn.ModuleList(
#             [nn.Conv2d(config.cnns_channels, config.cnns_channels, kernel_size=(1, 1)) for _ in range(config.cnns_numbers)])  # 串行点卷积
#         self.cnns_end = nn.Conv2d(config.cnns_channels, config.bert_embedding_dim, kernel_size=(1, 1))
#
#         self.bert_self_attention = BertSelfAttention(self.bert.config)
#         self.bert_pooler = BertPooler(self.bert.config)
#
#         self.gate = nn.Sequential(
#             nn.Linear(config.gate_size, config.gate_size),
#             nn.Sigmoid()
#         )
#
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, config.out_channels, (k, config.kernel_size)) for k in config.filter_sizes])  # 并行卷积
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(config.dropout)
#         self.cat_linear = nn.Linear(config.bert_embedding_dim * 2, config.bert_embedding_dim)
#
#         self.classifier3 = nn.Linear(config.classifier_size, config.class_num)
#         self.lsr = CrossEntropyLoss_LSR(config.device)
#         if config.use_bce_loss:
#             self.classifier = nn.Linear(config.classifier_size, 1)
#             self.loss_fun = nn.BCEWithLogitsLoss()
#         else:
#             self.classifier = nn.Linear(config.classifier_size, config.class_num)
#             self.loss_fun = nn.CrossEntropyLoss()
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x).squeeze(3))
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#     def _getMask(self, input, mask_idx):
#         '''
#         :param input:  shape:b,n,d
#         :param mask_idx: b
#         :return: b,d
#         '''
#         new_input = None
#         for idx in range(input.size(0)):
#             if new_input == None:
#                 new_input = input[idx][mask_idx[idx]].unsqueeze(dim=0)
#             else:
#                 new_input = torch.cat((new_input, input[idx][mask_idx[idx]].unsqueeze(dim=0)), dim=0)
#         return new_input
#
#     def forward(self, batch):
#         text_ids = batch['text_ids'].to(self.config.device)
#         label = batch['label'].to(self.config.device)
#         mask = batch['mask'].to(self.config.device)
#
#         encoder_out, text_cls = self.bert(text_ids, attention_mask=mask)
#
#         # serial cnn 串行卷积
#         cnn_first_out = self.relu(self.cnns_first(encoder_out.unsqueeze(3).permute(0, 2, 1, 3)))  # (b,n,d) -> (b,n,d,1) -> (b,d,n,1) -> (b,c,n,1)
#         if len(self.cnns):
#             cnns_out = cnn_first_out
#             for cnn in self.cnns:
#                 cnns_out = self.relu(cnn(cnns_out))
#             cnn_end_out = self.relu(self.cnns_end(cnns_out)).permute(0, 2, 1, 3).squeeze(3)  # (b,c,n,1) -> (b,d,n,1) -> (b,n,d,1)
#         else:
#             cnn_end_out = self.relu(self.cnns_end(cnn_first_out)).permute(0, 2, 1, 3).squeeze(3)
#         serial_cnn_out = self.dropout(cnn_end_out)
#
#         # 注意力， 池化， Gate融合
#         cat_out = self.cat_linear(torch.cat([encoder_out, serial_cnn_out], -1))
#         bert_self_attention_out = self.dropout(self.bert_self_attention(cat_out)[0])
#         bert_pool = self.bert_pooler(bert_self_attention_out)
#         average_pool = bert_self_attention_out.mean(dim=1)
#         gate_out = self.gate(torch.cat([bert_pool, average_pool], -1))
#
#         # 并行卷积
#         cnn_out = torch.cat([self.conv_and_pool(encoder_out.unsqueeze(1), conv) for conv in self.convs], 1)
#
#         out = torch.cat([gate_out, cnn_out], 1)
#         pre = self.classifier(out).squeeze(1)
#         loss = self.loss_fun(pre, label.squeeze(1))
#
#         return loss, pre

