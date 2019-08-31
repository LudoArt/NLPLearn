import codecs
import collections
from operator import itemgetter

RAW_DATA_EN = "H:/NLPLearn/Seq2Seq/data/train.en"  # 训练集数据文件
VOCAB_EN_OUTPUT = "H:/NLPLearn/Seq2Seq/data/en.vocab"  # 输出的词汇表文件
VOCAB_LENGTH_EN = 10000
RAW_DATA_ZH = "H:/NLPLearn/Seq2Seq/data/train.zh"  # 训练集数据文件
VOCAB_ZH_OUTPUT = "H:/NLPLearn/Seq2Seq/data/zh.vocab"  # 输出的词汇表文件
VOCAB_LENGTH_ZH = 4000


# 构建单词表
def build_vocab(RAW_DATA, VOCAB_OUTPUT, VOCAB_LENGTH):
    counter = collections.Counter()  # 统计单词出现频率
    with codecs.open(RAW_DATA, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 将"<unk>", "<sos>", "<eos>"加入词汇表
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_LENGTH:
        sorted_words = sorted_words[:VOCAB_LENGTH]

    with codecs.open(VOCAB_OUTPUT, "w", "utf-8") as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")


if __name__ == '__main__':
    build_vocab(RAW_DATA_EN, VOCAB_EN_OUTPUT, VOCAB_LENGTH_EN)
    build_vocab(RAW_DATA_ZH, VOCAB_ZH_OUTPUT, VOCAB_LENGTH_ZH)
