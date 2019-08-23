import codecs
import collections
from operator import itemgetter

RAW_DATA = "I:/NLPLearn/RNN/data/ptb.train.txt"  # 训练集数据文件
VOCAB_OUTPUT = "I:/NLPLearn/RNN/data/ptb.vocab"  # 输出的词汇表文件

counter = collections.Counter()  # 统计单词出现频率
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# 将句子结束符"<eos>"加入词汇表
sorted_words = ["<eos>"] + sorted_words

# 除了"<eos>"，还需要将"<unk>"和句子起始符"<sos>"加入词汇表，并从词汇表中删除低频词汇。
# 在PTB数据中，因为输入数据已经将低频词汇替换成了"<unk>"，因此不需要这一步
# sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
# if len(sorted_words) > 10000:
#     sorted_words = sorted_words[:10000]

with codecs.open(VOCAB_OUTPUT, "w", "utf-8") as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")
