import sys
import codecs

RAW_DATA_EN = "H:/NLPLearn/Seq2Seq/data/train.en"  # 原始的训练集数据文件
VOCAB_EN = "H:/NLPLearn/Seq2Seq/data/en.vocab"  # 生成的词汇表文件
OUTPUT_DATA_EN = "H:/NLPLearn/Seq2Seq/data/en.train"  # 将单词替换为单词编号后输出文件

RAW_DATA_ZH = "H:/NLPLearn/Seq2Seq/data/train.zh"  # 原始的训练集数据文件
VOCAB_ZH = "H:/NLPLearn/Seq2Seq/data/zh.vocab"  # 生成的词汇表文件
OUTPUT_DATA_ZH = "H:/NLPLearn/Seq2Seq/data/zh.train"  # 将单词替换为单词编号后输出文件


def word_to_id(RAW_DATA, VOCAB, OUTPUT_DATA):
    # 读取词汇表，并建立词汇到单词编号的映射
    with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了被删除的低频词，则替换为"<unk>"
    def get_id(word):
        if word in word_to_id:
            return word_to_id[word]
        else:
            return word_to_id['<unk>']

    fin = codecs.open(RAW_DATA, "r", "utf-8")
    fout = codecs.open(OUTPUT_DATA, "w", "utf-8")
    for line in fin:
        words = line.strip().split() + ["<eos>"]  # 读取单词并添加"<eos>"结束符
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)

    fin.close()
    fout.close()


if __name__ == '__main__':
    word_to_id(RAW_DATA_EN, VOCAB_EN, OUTPUT_DATA_EN)
    word_to_id(RAW_DATA_ZH, VOCAB_ZH, OUTPUT_DATA_ZH)
