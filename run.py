import torch
import os
import numpy as np
import random
from utils import build_dataloader
from train_test import train_model, to_test_model, init_network
from importlib import import_module


if __name__ == '__main__':
    bert = 'bert'

    x = import_module('models.' + bert)  # 决定训练哪个模型
    config = x.Config()  # 获得参数

    SEED = config.SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # np.save(config.seed_save_path, np.random.get_state())  # 保存随机种子
    # np.random.set_state(np.load(config.seed_save_path + '.npy', allow_pickle=True))  # 加载随机种子

    train_dataloader, dev_dataloader, test_dataloader = build_dataloader(config)  # 准备数据

    # 准备模型
    model = x.TextModel(config).to(config.device)
    init_network(model)

    # 训练, 保存, 测试
    if not os.path.exists(config.model_save_path):
        train_model(config, model, train_dataloader, dev_dataloader, test_dataloader)
    else:
        model.load_state_dict(torch.load(config.model_save_path))
        to_test_model(config, model, test_dataloader)

