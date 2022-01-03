import logging

import torch
from torch.utils.data import DataLoader, random_split


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict


class DataGenerator:
    def __init__(self, config, logger):
        self.vocab = []
        self.data = []
        self.seq = []
        self.split = ',.!?，。！？'
        self.config = config
        self.logger = logger
        self.load(config['data_path'])

    def str2list(self, data, en_kind=True):
        if en_kind:
            data = data.split(' ')
            en_real = []
            for word in data:
                if word[-1] in self.split:  # 切分词和标点符号
                    en_real.append(word[:-1])
                    en_real.append(word[-1])
                else:
                    en_real.append(word)
            return en_real
        else:
            data = list(data)
            return data

    def data2vocab(self):
        for i in range(len(self.data)):  # 遍历数据长度
            for kind in range(len(self.data[0])):  # 遍历语种长度
                for item in self.data[i][kind]:  # 遍历内容
                    if item not in self.vocab:
                        self.vocab.append(item)
        writer = open(self.config['vocab_path'], 'w', encoding='utf8')
        for item in ['[PAD]', '[BOS]', '[EOS]', '[UNK]']:
            writer.write(item + '\n')
        for item in self.vocab:
            writer.write(item + '\n')
        writer.close()

    def padding(self, lis, max_length):
        lis = lis[:max_length]
        lis += [self.vocab["[PAD]"]] * (max_length - len(lis))
        return lis

    def encode_sentence(self, lis, max_length, cls=True, sep=True):
        result = []
        if cls:
            result.append(self.vocab['[BOS]'])
        for item in lis:
            result.append(self.vocab.get(item, self.vocab['[UNK]']))
        if sep:
            result.append(self.vocab['[EOS]'])
        result = self.padding(result, max_length)
        return result

    def data2seq(self):
        for en, cn in self.data:
            # 如果要换语言就把下面的en、cn替换
            input_seq = self.encode_sentence(en, self.config['input_max_length'])
            output_seq = self.encode_sentence(cn, self.config['output_max_length'], True, False)
            gold = self.encode_sentence(cn, self.config['output_max_length'], False, True)
            self.seq.append([torch.LongTensor(input_seq),
                             torch.LongTensor(output_seq),
                             torch.LongTensor(gold)])
        self.seq = self.seq[:5000]  # 测试用只用100个

    def load_pred(self, input):
        # print(input)
        input_list = self.str2list(input)
        # print(input_list)
        input_seq = self.encode_sentence(input_list, self.config['input_max_length'])
        return torch.LongTensor(input_seq)

    def load(self, path):
        with open(path, encoding='utf8') as lines:
            for line in lines:
                line = line.split('\t')
                en = line[0]
                en = self.str2list(en)
                cn = line[1]
                cn = self.str2list(cn, en_kind=False)
                self.data.append([en, cn])
        self.logger.info('总数据量：%d' % len(self.data))
        self.data2vocab()
        self.vocab = load_vocab(self.config['vocab_path'])
        self.data2seq()
        self.logger.info('数据加载完毕')

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index]


def load_data(config, logger):
    dg = DataGenerator(config, logger)
    train_size = int(0.8 * len(dg))
    test_size = len(dg) - train_size
    train_dataset, test_dataset = random_split(dg, [train_size, test_size])
    train_dataset = DataLoader(train_dataset, batch_size=config['batch_size'])
    test_dataset = DataLoader(test_dataset, batch_size=config['batch_size'])
    return train_dataset, test_dataset


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from config import config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    config['data_path'] = '../data/cmn.txt'
    config['vocab_path'] = '../data/vocab.txt'

    # # 检查数据样式
    # train_dataset, test_dataset = load_data(config, logger)
    # for dataset in [train_dataset, test_dataset]:
    #     for x, y, gold in dataset:
    #         break
    #     print(len(dataset))
    #     print(x.shape, y.shape, gold.shape)

    # 测试独立一个句子编码
    dg = DataGenerator(config, logger)
    input_seq = dg.load_pred('hello hello.')
