import nltk

EN_FILE_PATH = "H:/NLPLearn/Seq2Seq/data/train.tags.en-zh.en"
EN_PARTICIPLE_OUTPUT = "H:/NLPLearn/Seq2Seq/data/train.en"
ZH_FILE_PATH = "H:/NLPLearn/Seq2Seq/data/train.tags.en-zh.zh"
ZH_PARTICIPLE_OUTPUT = "H:/NLPLearn/Seq2Seq/data/train.zh"


# 英文分词处理
def english_participle(FILE_PATH, PARTICIPLE_OUTPUT):
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = []
        for line in f.readlines():
            text.append(line)

    # sent_tokenize 文本分句处理，text是一个英文句子或文章
    with open(PARTICIPLE_OUTPUT, 'w', encoding='utf-8') as f:
        for line in text:
            # word_tokenize 分词处理，分词不支持中文
            words = nltk.word_tokenize(line)
            for word in words:
                f.write(word + ' ')
            f.write('\n')


# 中文分词处理
def chinese_participle(FILE_PATH, PARTICIPLE_OUTPUT):
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = []
        for line in f.readlines():
            text.append(line)

    # sent_tokenize 文本分句处理，text是一个英文句子或文章
    with open(PARTICIPLE_OUTPUT, 'w', encoding='utf-8') as f:
        for line in text:
            for n, char in enumerate(line):
                f.write(char + ' ')


if __name__ == '__main__':
    chinese_participle(ZH_FILE_PATH, ZH_PARTICIPLE_OUTPUT)

