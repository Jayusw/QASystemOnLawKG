import spacy
import pyltp
# 加载中文自然语言处理工具包
segmentor = pyltp.Segmentor()
segmentor.load('./ltp_data_v3.4.0/cws.model')
postagger = pyltp.Postagger()
postagger.load('./ltp_data_v3.4.0/pos.model')
# 加载spaCy中文模型
nlp = spacy.load('zh_core_web_sm')
# 输入文本
# with open('question.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
text = "根据《中华人民共和国刑法》第一百二十三条规定，饮酒后驾驶机动车的，处五日以上十日以下拘留，并处一千元以上二千元以下罚款。"
# 分句
sentences = pyltp.SentenceSplitter.split(text)
# 实体识别
for sentence in sentences:
    # 分词和词性标注
    words = segmentor.segment(sentence)
    postags = postagger.postag(words)
    # 构建spaCy识别实体所需的Doc对象
    doc = nlp(' '.join(words))
    # 提取实体
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['LAW', 'LAW_TERMS']:
            entities.append(ent.text)
    # 输出实体
    if len(entities) > 0:
        print('实体：', entities)
# 释放资源
segmentor.release()
postagger.release()