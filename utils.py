import torch
import re
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def process_data(config):
    filters1 = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?',
               '@', '\[', '\\', '\\\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '\x07']
    filters2 = ['#', '$', '%', '&', '\+', '/', '<', '=', '>', '@', '^', '_', '\|',
                '\\', '\\\\', '\t', '\n', '\x97', '\x96', '\x07']
    filters3 = ['\\\\', '\x07']

    data = pd.read_csv(config.file_path)  # 获取基本数据
    all_text = data['Text'].values
    course_type = data['CourseType'].values
    course_display_name = data['course_display_name'].values
    all_label = data['Urgency(1-7)'].values

    def create_csv(save_path, input_text, input_label):
        data_csv = {'Text': input_text, 'Label': input_label}
        df = pd.DataFrame(data_csv)
        df.to_csv(save_path, index=False)

    # Group A
    if not os.path.exists('data/groupA_train.csv'):
        a_train_text, a_test_text, a_train_label, a_test_label = train_test_split(all_text, all_label, test_size=config.test_size, stratify=all_label)
        a_test_text, a_dev_text, a_test_label, a_dev_label = train_test_split(a_test_text, a_test_label, test_size=config.dev_size, stratify=a_test_label)
        create_csv('data/groupA_train.csv', a_train_text, a_train_label)
        create_csv('data/groupA_test.csv', a_test_text, a_test_label)
        create_csv('data/groupA_dev.csv', a_dev_text, a_dev_label)
    else:
        a_train_text = pd.read_csv('data/groupA_train.csv')['Text']
        a_train_label = pd.read_csv('data/groupA_train.csv')['Label']
        a_test_text = pd.read_csv('data/groupA_test.csv')['Text']
        a_test_label = pd.read_csv('data/groupA_test.csv')['Label']
        a_dev_text = pd.read_csv('data/groupA_dev.csv')['Text']
        a_dev_label = pd.read_csv('data/groupA_dev.csv')['Label']

    # Group B
    if not os.path.exists('data/groupB_train.csv'):
        b_train_text = []
        b_train_label = []
        b_test_text = []
        b_test_label = []
        b_test_spilt = ['HumanitiesScience/StatLearning/Winter2014', 'Medicine/HRP258/Statistics_in_Medicine',
                        'Medicine/SURG210/Managing_Emergencies_What_Every_Doctor_Must_Know']
        education_text = []
        education_label = []
        for i in range(len(all_label)):
            b_text = all_text[i]
            b_label = all_label[i]
            b_display = course_display_name[i]
            c_type = course_type[i]
            if c_type == 'Education':
                education_text.append(b_text)
                education_label.append(b_label)
            else:
                if b_display in b_test_spilt:
                    b_test_text.append(b_text)
                    b_test_label.append(b_label)
                else:
                    b_train_text.append(b_text)
                    b_train_label.append(b_label)
        education_train_text, education_test_text, education_train_label, education_test_label = train_test_split(
            education_text, education_label, test_size=config.test_size, stratify=education_label)
        b_train_text.extend(education_train_text)
        b_train_label.extend(education_train_label)
        b_test_text.extend(education_test_text)
        b_test_label.extend(education_test_label)
        b_test_text, b_dev_text, b_test_label, b_dev_label = train_test_split(b_test_text, b_test_label, test_size=config.dev_size, stratify=b_test_label)
        create_csv('data/groupB_train.csv', b_train_text, b_train_label)
        create_csv('data/groupB_test.csv', b_test_text, b_test_label)
        create_csv('data/groupB_dev.csv', b_dev_text, b_dev_label)
    else:
        b_train_text = pd.read_csv('data/groupB_train.csv')['Text']
        b_train_label = pd.read_csv('data/groupB_train.csv')['Label']
        b_test_text = pd.read_csv('data/groupB_test.csv')['Text']
        b_test_label = pd.read_csv('data/groupB_test.csv')['Label']
        b_dev_text = pd.read_csv('data/groupB_dev.csv')['Text']
        b_dev_label = pd.read_csv('data/groupB_dev.csv')['Label']

    # Group C
    if not os.path.exists('data/groupC_train.csv'):
        c_train_text = []
        c_train_label = []
        c_test_text = []
        c_test_label = []
        c_test_spilt = ['Humanities']
        for i in range(len(all_label)):
            c_text = all_text[i]
            c_label = all_label[i]
            c_type = course_type[i]
            if c_type in c_test_spilt:
                c_test_text.append(c_text)
                c_test_label.append(c_label)
            else:
                c_train_text.append(c_text)
                c_train_label.append(c_label)
        c_test_text, c_dev_text, c_test_label, c_dev_label = train_test_split(c_test_text, c_test_label, test_size=config.dev_size, stratify=c_test_label)
        create_csv('data/groupC_train.csv', c_train_text, c_train_label)
        create_csv('data/groupC_test.csv', c_test_text, c_test_label)
        create_csv('data/groupC_dev.csv', c_dev_text, c_dev_label)
    else:
        c_train_text = pd.read_csv('data/groupC_train.csv')['Text']
        c_train_label = pd.read_csv('data/groupC_train.csv')['Label']
        c_test_text = pd.read_csv('data/groupC_test.csv')['Text']
        c_test_label = pd.read_csv('data/groupC_test.csv')['Label']
        c_dev_text = pd.read_csv('data/groupC_dev.csv')['Text']
        c_dev_label = pd.read_csv('data/groupC_dev.csv')['Label']

    def build_data(input_text, input_label):
        output_data = []
        for i in tqdm(range(len(input_text))):
            text = input_text[i]
            label = 0 if input_label[i] < 4 else 1

            text_filter = re.sub('|'.join(filters3), ' ', text)  # 过滤符合
            text_tokenizer = config.tokenizer(text_filter)
            text_ids = text_tokenizer['input_ids']
            attention_mask = text_tokenizer['attention_mask']
            mask_idx = 0

            if not config.collate_length_processing:  # collate_length_processing默认为False
                if len(text_ids) < config.max_len:
                    attention_mask = attention_mask + [0] * (config.max_len - len(text_ids))
                    text_ids = text_ids + [0] * (config.max_len - len(text_ids))  # 对长度不够的序列用0进行填充
                else:
                    attention_mask = [1] * config.max_len
                    text_ids = text_ids[:config.max_len]

                text_ids = torch.LongTensor(text_ids)
                if config.use_bce_loss:
                    label = torch.FloatTensor([label])
                else:
                    label = torch.LongTensor([label])
                attention_mask = torch.LongTensor(attention_mask)
                mask_idx = torch.LongTensor(mask_idx)
                data_ = {
                    'text_ids': text_ids,
                    'label': label,
                    'mask': attention_mask,
                    'mask_idx': mask_idx
                }
                output_data.append(data_)
            else:
                data_ = {
                    'text_ids': text_ids,
                    'label': label,
                    'mask': attention_mask,
                }
                # if len(text_ids) <= 128:
                #     output_data.insert(0, data_)
                # else:
                #     output_data.insert(len(output_data), data_)

                # if i == 0:  # 使文本按文本长度从小到大排列
                #     output_data.insert(0, data_)
                # else:
                #     for j in range(len(output_data)):
                #         text_ids_len = len(text_ids)
                #         contrast_ids_len = len(output_data[j]['text_ids'])
                #         if text_ids_len <= contrast_ids_len:
                #             output_data.insert(j, data_)  # 放在第j个元素前面
                #             break
                #         elif j == len(output_data) - 1:
                #             output_data.insert(j + 1, data_)  # 放在最后一个元素后面
        return output_data

    def reconstruction_data(input_text, input_label):
        output_data = []
        prompt = {
            1: 'this is a [MASK] comment for [CLS]',
            2: 'the [CLS] gets a [MASK] comment',
            3: 'the [CLS] of the review is [MASK]',
            4: 'that gets a [MASK] comment [SEP] [CLS] ',
            5: 'whose comment is [MASK] [SEP] [CLS]',
            6: 'with [MASK] comment [SEP] [CLS] '
        }

        for i in tqdm(range(len(input_text))):
            text = input_text[i]
            label = 0 if input_label[i] < 4 else 1

            if label == 1:
                text = text + prompt[0]

            text_filter = re.sub('|'.join(filters3), ' ', text)  # 过滤符合
            text_tokenizer = config.tokenizer(text_filter)
            text_ids = text_tokenizer['input_ids']
            attention_mask = text_tokenizer['attention_mask']

            if len(text_ids) < config.max_len:
                attention_mask = attention_mask + [0] * (config.max_len - len(text_ids))
                text_ids = text_ids + [0] * (config.max_len - len(text_ids))  # 对长度不够的序列用0进行填充
            else:
                attention_mask = [1] * config.max_len
                text_ids = text_ids[:config.max_len]

            text_ids = torch.LongTensor(text_ids)
            if config.use_bce_loss:
                label = torch.FloatTensor([label])
            else:
                label = torch.LongTensor([label])
            attention_mask = torch.LongTensor(attention_mask)
            data_ = {
                'text_ids': text_ids,
                'label': label,
                'mask': attention_mask,
            }
            output_data.append(data_)
        return output_data

    if config.GroupA:
        train_text = a_train_text
        train_label = a_train_label
        test_text = a_test_text
        test_label = a_test_label
        dev_text = a_dev_text
        dev_label = a_dev_label
    elif config.GroupB:
        train_text = b_train_text
        train_label = b_train_label
        test_text = b_test_text
        test_label = b_test_label
        dev_text = b_dev_text
        dev_label = b_dev_label
    else:
        train_text = c_train_text
        train_label = c_train_label
        test_text = c_test_text
        test_label = c_test_label
        dev_text = c_dev_text
        dev_label = c_dev_label

    if config.re_data:
        print("----------开始重构训练集----------")
        train_build_data = reconstruction_data(train_text, train_label)
    else:
        print("----------准备数据----------")
        train_build_data = build_data(train_text, train_label)
    test_build_data = build_data(test_text, test_label)  # test_text, test_label
    dev_build_data = build_data(dev_text, dev_label)  # dev_text, dev_label
    return train_build_data, test_build_data, dev_build_data


class MyDataset(Dataset):
    def __init__(self, all_data, config):
        self.all_data = all_data
        self.config = config

    def __getitem__(self, index):
        if not self.config.collate_length_processing:  # collate_length_processing默认为False
            return self.all_data[index]
        else:
            text = self.all_data[index]['text_ids']
            label = self.all_data[index]['label']
            mask = self.all_data[index]['mask']
            return text, label, mask

    def __len__(self):
        return len(self.all_data)


def collate_fn(batch):
    """
    使最大输入文本长度为一个batch_size内的最大文本长度（不同的batch_size文本长度可以不同）
    """
    batch_text = []
    batch_label = []
    batch_mask = []
    text, label, mask = zip(*batch)

    max_len = min(max([len(s) for s in text]), 200)

    for i in range(len(text)):
        _text = text[i]
        _label = label[i]
        _mask = mask[i]
        if len(_text) < max_len:
            _mask = _mask + [0] * (max_len - len(_text))
            _text = _text + [0] * (max_len - len(_text))
        else:
            _text = _text[:max_len]
            _mask = _mask[:max_len]

        batch_text.append(_text)
        batch_label.append([_label])
        batch_mask.append(_mask)
    batch_text = torch.LongTensor(tuple(batch_text))
    batch_label = torch.FloatTensor(tuple(batch_label))
    batch_mask = torch.LongTensor(tuple(batch_mask))
    return {'text_ids': batch_text, 'label': batch_label, 'mask': batch_mask}


def build_dataloader(config):
    train_data, test_data, dev_data = process_data(config)

    train_dataset = MyDataset(train_data, config)
    dev_dataset = MyDataset(dev_data, config)
    test_dataset = MyDataset(test_data, config)

    if not config.collate_length_processing:  # collate_length_processing默认为False
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader


# def collate_fn(self, batch):
    #     text = [x[0] for x in batch]
    #     print(text)
    #     mask = [x[2] for x in batch]
    #     label = [x[1] for x in batch]
    #
    #     batch_len = len(text)
    #
    #     max_len = max([len(s) for s in text])
    #     max_len = min(max_len, 512)
    #
    #     batch_text = 0 * torch.ones((batch_len, max_len))
    #     batch_mask = 0 * torch.ones(batch_len, max_len)
    #
    #     for i in range(len(text)):
    #         cur_len = len(text[i])
    #
    #         batch_text[i][:cur_len] = text[i]
    #         batch_mask[i][:cur_len] = mask[i]
    #
    #     batch_text = torch.tensor(batch_text, dtype=torch.long)
    #     batch_label = torch.tensor(label, dtype=torch.float)
    #     batch_mask = torch.tensor(batch_mask, dtype=torch.long)
    #
    #     batch_text = batch_text.cuda()
    #     batch_label = batch_label.cuda()
    #     batch_mask = batch_mask.cuda()
    #
    #
    #     return batch_text, batch_mask, batch_label
