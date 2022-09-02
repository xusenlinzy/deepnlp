import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    文本卷积网络
    + 📖 首先通过卷积核提取多种`n-gram`特征
    + 📖 最后通过最大池化提取文本主要特征
    
    Args:
        `num_labels`: 类别数
        `n_filters`: 卷积核个数（对应2维卷积的通道数）
        `filter_sizes`: 卷积核的尺寸(3, 4, 5)
    
    Reference:
        ⭐️ [Convolutional Neural Networks for Sentence Classification.](https://arxiv.org/abs/1408.5882)
        🚀 [Code](https://github.com/yoonkim/CNN_sentence)
    """

    def __init__(self, num_labels, hidden_size, n_filters=3, filter_sizes=(3, 4, 5)):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters,
                       kernel_size=(fs, hidden_size)) for fs in filter_sizes])
        self.classifier = nn.Linear(len(filter_sizes) * n_filters, num_labels)

    def forward(self, sequence_output):
        # sequence_output = [batch size, 1, sent len, emb dim]
        sequence_output = sequence_output.unsqueeze(1).float()
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        conved = [F.relu(conv(sequence_output)).squeeze(3) for conv in self.convs]
        # pooled_n = [batch size, n_filters]
        pooled = [F.max_pool1d(conv, int(conv.shape[2])).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)  # [batch size, n_filters * len(filter_sizes)]
        return self.classifier(cat)


class DPCNN(nn.Module):
    """
    金字塔结构的深度卷积网络
    + 📖 `Region embedding`提取文本的`n-gram`特征
    + 📖 等长卷积+1/2池化层获取更长的上下文信息
    + 📖 残差连接避免网络退化，使得深层网络更容易训练
    
    Args:
        `num_labels`: 类别数
        `n_filters`: 卷积核个数（对应2维卷积的通道数）
    
    Reference:
        ⭐️ [Deep Pyramid Convolutional Neural Networks for Text Categorization.](https://aclanthology.org/P17-1052.pdf)
        🚀 [Code](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
    """

    def __init__(self, num_labels, hidden_size, n_filters=3):
        super(DPCNN, self).__init__()
        self.conv_region = nn.Conv2d(1, n_filters, (3, hidden_size), stride=1)
        self.conv = nn.Conv2d(n_filters, n_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.relu = nn.ReLU()
        self.classifier = nn.Linear(n_filters, num_labels)

    def forward(self, sequence_output):
        # [batch size, 1, seq len, emb dim]
        x = sequence_output.unsqueeze(1).to(torch.float32)
        # [batch size, n_filters, seq len-3+1, 1]
        x = self.conv_region(x)
        # [batch size, n_filters, seq len, 1]
        x = self.padding1(x)
        x = self.relu(x)
        # [batch size, n_filters, seq len-3+1, 1]
        x = self.conv(x)
        # [batch size, n_filters, seq len, 1]
        x = self.padding1(x)
        x = self.relu(x)
        # [batch size, n_filters, seq len-3+1, 1]
        x = self.conv(x)
        # [batch size, n_filters, 1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        # [batch size, n_filters]
        x = x.squeeze()
        return self.classifier(x)

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


class RCNN(nn.Module):
    """
    循环卷积网络
    + 📖 `RNN`擅长处理序列结构，能够考虑到句子的上下文信息，但是一个个句子中越往后的词重要性越高，有可能影响最后的分类结果
    + 📖 `CNN`能够通过最大池化获得最重要的特征，但是滑动窗口大小不容易确定，选的过小容易造成重要信息丢失
    + 📖 首先用双向循环结构获取上下文信息，其次使用最大池化层获取文本的重要特征
    
    Args:
        `num_labels`: 类别数目
        `hidden_size`: 词嵌入维度
        `rnn_dim`: `rnn`隐藏层的维度
        `rnn_layers`: `rnn`层数
        `rnn_type`: `rnn`类型，包括[`lstm`, `gru`, `rnn`]
        `bidirectional`: 是否双向
        `dropout`: `dropout`概率
        `batch_first`: 第一个维度是否是批量大小
    
    Reference:
        ⭐️ [Recurrent convolutional neural networks for text classification.](https://dl.acm.org/doi/10.5555/2886521.2886636)
        🚀 [Code](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
    """

    def __init__(self, num_labels, hidden_size, rnn_dim, rnn_type='lstm', rnn_layers=1, bidirectional=True,
                 batch_first=True, dropout=0.2):
        super(RCNN, self).__init__()
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size,
                               rnn_dim,
                               num_layers=rnn_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size,
                              rnn_dim,
                              num_layers=rnn_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(hidden_size,
                              rnn_dim,
                              num_layers=rnn_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        # 1 x 1 卷积等价于全连接层，故此处使用全连接层代替
        self.fc_cat = nn.Linear(rnn_dim * 2 + hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, sequence_output):
        output, _ = self.rnn(sequence_output)
        seq_len = output.shape[1]
        # 拼接左右上下文信息
        output = torch.tanh(self.fc_cat(torch.cat((output, sequence_output), dim=2)))
        output = torch.transpose(output, 1, 2)
        # 最大池化
        output = F.max_pool1d(output, int(seq_len)).squeeze().contiguous()
        return self.classifier(output)


class TextRNN(nn.Module):
    """
    循环神经网络
    
    Args:
        `num_labels`: 类别数目
        `hidden_size`: 词嵌入维度
        `rnn_dim`: `rnn`隐藏层的维度
        `rnn_layers`: `rnn`层数
        `rnn_type`: `rnn`类型，包括[`lstm`, `gru`, `rnn`]
        `bidirectional`: 是否双向
        `dropout`: `dropout`概率
        `batch_first`: 第一个维度是否是批量大小
        `attention`: 是否增加注意力机制
    
    Reference:
        🚀 [Code](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
    """

    def __init__(self, num_labels, hidden_size, rnn_dim, rnn_type='lstm', rnn_layers=1, bidirectional=True,
                 batch_first=True, dropout=0.2, attention=True):
        super(TextRNN, self).__init__()
        self.attention = attention
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size,
                               rnn_dim,
                               num_layers=rnn_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size,
                              rnn_dim,
                              num_layers=rnn_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(hidden_size,
                              rnn_dim,
                              num_layers=rnn_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        # query向量
        if self.attention:
            self.u = nn.Parameter(torch.randn(rnn_dim * 2), requires_grad=True)
            self.tanh = nn.Tanh()

        self.fc = nn.Linear(rnn_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_output):
        output, _ = self.rnn(sequence_output)

        if self.attention:
            alpha = F.softmax(torch.matmul(self.tanh(output), self.u), dim=1).unsqueeze(-1)
            output = output * alpha  # [batch_size, seq_len, rnn_dim * num_directionns ]
            output = torch.sum(output, dim=1)  # [batch_size, hidden_dim]
            output = self.dropout(output)
        return self.fc(output)
