from src.loader import load_data
from collections import defaultdict

from transformer.Translator import Translator


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.data = load_data(config, logger)
        self.reverse_vocab = dict([(y, x) for x, y in self.data.dataset.vocab.items()])
        self.translator = Translator(self.model,
                                     config["beam_size"],
                                     config["output_max_length"],
                                     config["pad_idx"],
                                     config["pad_idx"],
                                     config["start_idx"],
                                     config["end_idx"])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.model.cpu()
        self.stats_dict = defaultdict(int)  # 用于存储测试结果
        for index, batch_data in enumerate(self.data):
            input_seqs, target_seqs, gold = batch_data
            for input_seq in input_seqs:
                generate = self.translator.translate_sentence(input_seq.unsqueeze(0))
                print("输入：", self.decode_seq(input_seq, True))
                print("输出：", self.decode_seq(generate))
                break
        return

    def decode_seq(self, seq, en=False):
        if en:
            return " ".join([self.reverse_vocab[int(idx)] for idx in seq])
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])


if __name__ == "__main__":
    pass