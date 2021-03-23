import paddle
import paddlenlp
import numpy as np
from functools import partial

import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import MapDatasetWrapper

from utils import load_vocab, convert_example
class SelfDefinedDataset(paddle.io.Dataset):
    def __init__(self, data):
        super(SelfDefinedDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
        
    def get_labels(self):
        return ["0", "1"]

def txt_to_list(file_name):
    res_list = []
    for line in open(file_name):
        res_list.append(line.strip().split('\t'))
    return res_list

trainlst = txt_to_list('train.txt')
devlst = txt_to_list('dev.txt')
testlst = txt_to_list('test.txt')

train_ds, dev_ds, test_ds = SelfDefinedDataset.get_datasets([trainlst, devlst, testlst])
label_list = train_ds.get_labels()
print(label_list)
for i in range(10):
    print (train_ds[i]
           # 下载词汇表文件word_dict.txt，用于构造词-id映射关系。
!wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt

# 加载词表
vocab = load_vocab('./senta_word_dict.txt')

for k, v in vocab.items():
    print(k, v)
    break
           # Reads data and generates mini-batches.
def create_dataloader(dataset,
                      trans_function=None,
                      mode='train',
                      batch_size=1,
                      pad_token_id=0,
                      batchify_fn=None):
    if trans_function:
        dataset = dataset.apply(trans_function, lazy=True)

    # return_list 数据是否以list形式返回
    # collate_fn  指定如何将样本列表组合为mini-batch数据。传给它参数需要是一个callable对象，需要实现对组建的batch的处理逻辑，并返回每个batch的数据。在这里传入的是`prepare_input`函数，对产生的数据进行pad操作，并返回实际长度等。
    dataloader = paddle.io.DataLoader(
        dataset,
        return_list=True,
        batch_size=batch_size,
        collate_fn=batchify_fn)
        
    return dataloader

# python中的偏函数partial，把一个函数的某些参数固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。
trans_function = partial(
    convert_example,
    vocab=vocab,
    unk_token_id=vocab.get('[UNK]', 1),
    is_test=False)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab['[PAD]']),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]


train_loader = create_dataloader(
    train_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='train',
    batchify_fn=batchify_fn)
dev_loader = create_dataloader(
    dev_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='validation',
    batchify_fn=batchify_fn)
test_loader = create_dataloader(
    test_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='test',
    batchify_fn=batchify_fn)
           #搭建lastm模型
class LSTMModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)

        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)

        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)

        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # text shape: (batch_size, num_tokens)
        # print('input :', text.shape)
        
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # print('after word-embeding:', embedded_text.shape)

        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirectional' else 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # print('after lstm:', text_repr.shape)


        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # print('after Linear classifier:', fc_out.shape)

        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # print('output:', logits.shape)
        
        # probs 分类概率值
        probs = F.softmax(logits, axis=-1)
        # print('output probability:', probs.shape)
        return probs

model= LSTMModel(
        len(vocab),
        len(label_list),
        direction='bidirectional',
        padding_idx=vocab['[PAD]'])
model = paddle.Model(model)
           optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=5e-5)

loss = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

model.prepare(optimizer, loss, metric)
           # 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
           results = model.evaluate(dev_loader)
print("Finally test acc: %.5f" % results['acc'])
#预测
label_map = {0: '悲伤', 1: '积极'}
results = model.predict(test_loader, batch_size=128)[0]
predictions = []

for batch_probs in results:
    # 映射分类label
    idx = np.argmax(batch_probs, axis=-1)
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    predictions.extend(labels)

# 看看预测数据前5个样例分类结果
#for idx, data in enumerate(test_ds.data[:10]):
#    print('Data: {} \t Label: {}'.format(data[0], predictions[idx]))

for i in range(len(results[0])):
    sad_level=results[0][i][0]/(results[0][i][0]+results[0][i][1])
    happy_level=results[0][i][1]/(results[0][i][0]+results[0][i][1])
    popele_level=int(-((-sad_level+happy_level)/2-0.5)*10)
    #print(popele_level)
    print('Data: {} \t 网抑云等级: {}'.format(test_ds.data[i], popele_level))
                      
