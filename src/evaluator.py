import os
import matplotlib.pyplot as plt
from src.loader import load_data
from collections import defaultdict
from transformer.Translator import Translator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

"""
模型效果测试类
"""

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Evaluator:
    def __init__(self, config, model, logger, train_data, test_data):
        self.config = config
        self.model = model
        self.logger = logger
        self.data = test_data
        self.train_data = train_data
        self.reverse_vocab = dict([(y, x) for x, y in self.data.dataset.dataset.vocab.items()])
        self.translator = Translator(self.model,
                                     config['beam_size'],
                                     config['output_max_length'],
                                     config['pad_idx'],
                                     config['pad_idx'],
                                     config['start_idx'],
                                     config['end_idx'])

    def eval(self, epoch):
        self.logger.info('开始测试第%d轮模型效果：' % epoch)
        smooth = SmoothingFunction()
        self.model.eval()
        self.model.cpu()
        self.stats_dict = defaultdict(int)  # 用于存储测试结果
        score_avg = 0
        for index, batch_data in enumerate(self.data):
            input_seqs, target_seqs, gold = batch_data
            i = 0
            for input_seq, target_seq in zip(input_seqs, target_seqs):
                # 序列解码
                generate = self.translator.translate_sentence(input_seq.unsqueeze(0))
                input_sentence = self.decode_seq(input_seq, True)
                target_sentence = self.decode_seq(target_seq)
                generate_sentence = self.decode_seq(generate)
                # 计算bleu相似度
                score = sentence_bleu([target_sentence], generate_sentence, smoothing_function=smooth.method1)
                score_avg += score
                i += 1
                if i > 20:
                    break
            score_avg /= i
            # 打印
            print('---测试集---')
            print('输入：', input_sentence)
            print('标签：', target_sentence)
            print('输出：', generate_sentence)
            print('平均相似度：', score_avg)
            break
        return score_avg

    def eval_t(self, epoch):
        self.logger.info('开始测试第%d轮模型效果：' % epoch)
        smooth = SmoothingFunction()
        self.model.eval()
        self.model.cpu()
        self.stats_dict = defaultdict(int)  # 用于存储测试结果
        score_avg = 0
        for index, batch_data in enumerate(self.train_data):
            input_seqs, target_seqs, gold = batch_data
            i = 0
            for input_seq, target_seq in zip(input_seqs, target_seqs):
                # 序列解码
                generate = self.translator.translate_sentence(input_seq.unsqueeze(0))
                input_sentence = self.decode_seq(input_seq, True)
                target_sentence = self.decode_seq(target_seq)
                generate_sentence = self.decode_seq(generate)
                # 计算bleu相似度
                score = sentence_bleu([target_sentence], generate_sentence, smoothing_function=smooth.method1)
                score_avg += score
                i += 1
                if i > 20:
                    break
            score_avg /= i
            # 打印
            print('---训练集---')
            print('输入：', input_sentence)
            print('标签：', target_sentence)
            print('输出：', generate_sentence)
            print('平均相似度：', score_avg)
            break
        return score_avg

    def decode_seq(self, seq, en=False):
        # 修改bleu用的sentence备份
        sentence = [self.reverse_vocab[int(idx)] for idx in seq if idx not in [0, 1, 2]]
        if en:
            return " ".join(sentence)
        return "".join(sentence)

    def plot_and_save(self, epoch, t_s, score_avgs, losses):
        best_score = max(score_avgs)

        x = range(epoch)
        fig = plt.figure()
        plt.plot(x, t_s, label='train score avg')
        plt.plot(x, score_avgs, label='test score avg')
        plt.plot(x, losses, label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('num')
        plt.title('训练曲线 best eval=%f' % best_score)
        plt.legend()
        plt.savefig(os.path.join(self.config['model_path'],
                                 'report-%s-%f.png' % (self.config['learning_rate'], best_score)))


if __name__ == '__main__':
    pass
