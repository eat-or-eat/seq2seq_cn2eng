import os
import torch
import time
import logging
import redis

from config import config
from src.loader import DataGenerator
from transformer.Models import Transformer
from transformer.Translator import Translator

"""
模型预测类
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class eng2cn:
    def __init__(self, config, logger):
        self.redis_cache = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.config = config
        self.dg = DataGenerator(config, logger)
        self.reverse_vocab = dict([(y, x) for x, y in self.dg.vocab.items()])
        self.model = Transformer(config['vocab_size'], config['vocab_size'], 0, 0,
                                 d_word_vec=128, d_model=128, d_inner=256,
                                 n_layers=1, n_head=2, d_k=64, d_v=64,
                                 )
        self.model.load_state_dict(torch.load(os.path.join(self.config['model_path'],
                                                           'epoch_200.pth')))
        self.translator = Translator(self.model,
                                     config['beam_size'],
                                     config['output_max_length'],
                                     config['pad_idx'],
                                     config['pad_idx'],
                                     config['start_idx'],
                                     config['end_idx'])

    def decode_seq(self, seq, en=False):
        # 修改bleu用的sentence备份
        sentence = [self.reverse_vocab[int(idx)] for idx in seq if idx not in [0, 1, 2]]
        if en:
            return " ".join(sentence)
        return "".join(sentence)

    def predict(self, sentence, redis=False):
        if redis:
            from_cache = self.redis_cache.get(sentence)
            if from_cache is not None:
                # print('using redis')
                return from_cache
        input_seq = self.dg.load_pred(sentence)
        generate = self.translator.translate_sentence(input_seq.unsqueeze(0))
        generate_sentence = self.decode_seq(generate)
        self.redis_cache.set(sentence, generate_sentence, ex=100)
        return generate_sentence


if __name__ == '__main__':
    e2c = eng2cn(config, logger)

    start_time = time.time()
    for i in range(500):
        result = e2c.predict('I won!', redis=True)
    print('有redis，耗时：', time.time() - start_time)

    for i in range(500):
        result = e2c.predict('I won!')
    print('没有redis，耗时：', time.time() - start_time)

    result = e2c.predict("They won't find Tom .", redis=True)
    print(result)
