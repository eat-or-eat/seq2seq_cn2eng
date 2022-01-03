import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json
from config import config
from src.evaluator import Evaluator
from src.loader import load_data

# 这里用的是github大神开源的pytorch版本的transformer，不是pip安装的
from transformer.Models import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def choose_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def main(config):
    # 创建保存模型的目录
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])
    # 加载模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    model = Transformer(config['vocab_size'], config['vocab_size'], 0, 0,
                        d_word_vec=128, d_model=128, d_inner=256,
                        n_layers=1, n_head=2, d_k=64, d_v=64,
                        )
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info('gpu可以使用，迁移模型至gpu')
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data, test_data = load_data(config, logger)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, train_data, test_data)
    # 加载loss
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
    # 训练
    ts, score_avgs, losses = [], [], []
    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info('epoch %d begin' % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, target_seq, gold = batch_data
            pred = model(input_seq, target_seq)
            loss = loss_func(pred, gold.view(-1))
            train_loss.append(float(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info('epoch average loss: %f' % np.mean(train_loss))
        score_avg = evaluator.eval(epoch)
        t = evaluator.eval_t(epoch)
        score_avgs.append(score_avg*100)
        ts.append(t*100)
        losses.append(np.mean(train_loss))

    evaluator.plot_and_save(epoch, ts, score_avgs, losses)
    model_path = os.path.join(config['model_path'], 'epoch_%d.pth' % epoch)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == '__main__':
    for lr in [1e-3]:
        config['learning_rate'] = lr
        main(config)
