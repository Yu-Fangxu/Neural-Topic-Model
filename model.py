import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from torch.distributions import LogNormal, Dirichlet, kl_divergence
import torch.nn.functional as F
# dataset = "mr"
# BoW = np.load(f"temp/{dataset}.BoW.npy")
# BoW = BoW[:1000]
# print(BoW.shape)
news = fetch_20newsgroups(subset='all')
vectorizer = CountVectorizer(max_df=0.5, min_df=100, stop_words='english')
docs = torch.from_numpy(vectorizer.fit_transform(news['data']).toarray())

vocab = pd.DataFrame(columns=['word', 'index'])
vocab['word'] = vectorizer.get_feature_names()
vocab['index'] = vocab.index
# print('Dictionary size: %d' % len(vocab))
# print('Corpus size: {}'.format(docs.shape))
class PLSA:
    def __init__(self, X, K, words, iters=3):
        """
        :param X: word-doc矩阵
        :param K: 设定的topic数
        :param words: 单词列表
        :param iters: 迭代次数
        """

        self.N, self.M = X.shape

        self.X = X.T
        self.K = K
        self.iters = iters
        self.P_wi_zk = np.random.rand(self.K, self.M)
        self.P_zk_dj = np.random.rand(self.N, self.K)
        self.P_zk_wi_dj = np.zeros((self.M, self.N, self.K))
        for k in range(self.K):
            self.P_wi_zk[k] /= np.sum(self.P_wi_zk[k])

        for n in range(self.N):
            self.P_zk_dj[n] /= np.sum(self.P_zk_dj[n])

    def calc(self):
        for i in tqdm(range(self.iters)):
            for m in range(self.M):
                for n in range(self.N):
                    sums = 1e-4
                    for k in range(self.K):
                        self.P_zk_wi_dj[m, n, k] = self.P_wi_zk[k, m] * self.P_zk_dj[n, k]  #计算P(zk|wi,dj)的分子部分，即P(wi|zk)*P(zk|dj)
                        sums += self.P_zk_wi_dj[m, n, k]  #计算P(zk|wi,dj)的分母部分，即P(wi|zk)*P(zk|dj)在K个话题上的总和
                    self.P_zk_wi_dj[m, n, :] = self.P_zk_wi_dj[m, n, :] / sums  #得到单词-文本对(wi,dj)条件下的P(zk|wi,dj)
        #执行M步，计算P(wi|zk)
            for k in range(self.K):
                s1 = 1e-4
                for m in range(self.M):
                    self.P_wi_zk[k, m] = 0
                    for n in range(self.N):
                        self.P_wi_zk[k, m] += self.X[m, n] * self.P_zk_wi_dj[m, n, k]  #计算P(wi|zk)的分子部分，即n(wi,dj)*P(zk|wi,dj)在N个文本上的总和，其中n(wi,dj)为单词-文本矩阵X在文本对(wi,dj)处的频次
                    s1 += self.P_wi_zk[k, m]  #计算P(wi|zk)的分母部分，即n(wi,dj)*P(zk|wi,dj)在N个文本和M个单词上的总和
                self.P_wi_zk[k, :] = self.P_wi_zk[k, :] / s1  #得到话题zk条件下的P(wi|zk)
        #执行M步，计算P(zk|dj)
            for n in range(self.N):
                for k in range(self.K):
                    self.P_zk_dj[n, k] = 0
                    for m in range(self.M):
                        self.P_zk_dj[n, k] += self.X[m, n] * self.P_zk_wi_dj[m, n, k]  #同理计算P(zk|dj)的分子部分，即n(wi,dj)*P(zk|wi,dj)在N个文本上的总和
                    self.P_zk_dj[n, k] = self.P_zk_dj[n, k] / np.sum(self.X[:, n])  #得到文本dj条件下的P(zk|dj)，其中n(dj)为文本dj中的单词个数，由于我们只取了出现频次前1000的单词，所以这里n(dj)计算的是文本dj中在单词列表中的单词数
        return self.P_wi_zk, self.P_zk_dj

class NVDM(nn.Module):
    '''Implementation of the NVDM model as described in `Neural Variational Inference for
    Text Processing (Miao et al. 2016) <https://arxiv.org/pdf/1511.06038.pdf>`_.
    Args:
        vocab_size (int): The vocabulary size that will be used for the BOW's (how long the BOW
            vectors will be).
        num_topics (:obj:`int`, optional): Set to `100` by default. The number of latent topics
            to maintain. Corresponds to hidden vector dimensionality `K` in the technical writing.
        hidden_size(:obj:`int`, optional): Set to `256` by default. The number of hidden units to
            include in each layer of the multilayer perceptron (MLP).
        hidden_layers(:obj:`int`, optional): Set to `1` by default. The number of hidden layers to
            generate when creating the MLP component of the model.
        nonlinearity(:obj:`torch.nn.modules.activation.*`, optional): Set to
            :obj:`torch.nn.modules.activation.Tanh` by default. Controls which nonlinearity to use
            as the activation function in the MLP component of the model.
    '''
    @staticmethod
    def _param_initializer(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, vocab_size, num_topics=50, hidden_size=64, hidden_layers=1, nonlinearity=nn.Tanh):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        # First MLP layer compresses from vocab_size to hidden_size
        mlp_layers = [nn.Linear(vocab_size, hidden_size), nonlinearity()]
        # Remaining layers operate in dimension hidden_size
        for _ in range(hidden_layers - 1):
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            mlp_layers.append(nonlinearity())

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.apply(NVDM._param_initializer)

        # Create linear projections for Gaussian params (mean & sigma)
        self.mean = nn.Linear(hidden_size, num_topics)
        self.mean.apply(NVDM._param_initializer)

        # Custom initialization for log_sigma
        self.log_sigma = nn.Linear(hidden_size, num_topics)
        self.log_sigma.bias.data.zero_()
        self.log_sigma.weight.data.fill_(0.)

        self.dec_projection = nn.Linear(num_topics, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)

    def forward(self, input_bows):
        # Run BOW through MLP
        pi = self.mlp(input_bows)

        # Use this to get mean, log_sig for Gaussian
        mean = self.mean(pi)
        log_sigma = self.log_sigma(pi)

        # Calculate KLD
        kld = -0.5 * torch.sum(1 - torch.square(mean) +
                               (2 * log_sigma - torch.exp(2 * log_sigma)), 1)
        # kld = mask * kld  # mask paddings

        # Use Gaussian reparam. trick to sample from distribution defined by mu, sig
        # This provides a sample h_tm from posterior q(h_tm | V) (tm meaning topic model)
        epsilons = torch.normal(0, 1, size=(
            input_bows.size()[0], self.num_topics)).to(input_bows.device)
        sample = (torch.exp(log_sigma) * epsilons) + mean

        # Softmax to get p(v_i | h_tm), AKA probabilities of words given hidden state
        logits = self.log_softmax(self.dec_projection(sample))

        # Lowerbound on NVDM true loss, used for optimization
        rec_loss = -1 * torch.sum(logits * input_bows, 1)
        
        # loss_nvdm_lb = torch.mean(rec_loss + kld)
        # rec_loss = torch.sum(rec_loss. dim=1) / input_bows.shape[0]
        
        # rec_loss = torch.mean(rec_loss)
        # kld = torch.mean(kld)
        return sample, logits, kld, rec_loss# loss_nvdm_lb
        
class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics=50, hidden_size=100, hidden_layers=2, nonlinearity=nn.Softplus):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        # First MLP layer compresses from vocab_size to hidden_size
        mlp_layers = [nn.Linear(vocab_size, hidden_size), nonlinearity()]
        for _ in range(hidden_layers - 1):
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            mlp_layers.append(nonlinearity())

        self.mlp = nn.Sequential(*mlp_layers)
        # using Dirichlet distribution directly
        self.h2t = nn.ModuleList([nn.Linear(hidden_size, num_topics),
                                 nn.BatchNorm1d(num_topics)])
        self.mean = nn.Linear(hidden_size, num_topics)
                                  
        self.log_sigma = nn.Sequential(nn.Linear(hidden_size, num_topics),
                                       nn.BatchNorm1d(num_topics))
                                       
        
        self.dec_projection = nn.Linear(num_topics, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)
    def forward(self, input_bows):
        h = self.mlp(input_bows)
        alpha = self.h2t[1](self.h2t[0](h)).exp()
        posterior = Dirichlet(alpha.cpu())
        if self.training:
            sample = posterior.rsample().cuda()
        else:
            sample = posterior.mean().cuda()
        
        # mean = self.mean(h)
        # log_sigma = self.log_sigma(h)
        # epsilons = torch.normal(0, 1, size=(
        # input_bows.size()[0], self.num_topics)).to(input_bows.device)
        
        # sample = (torch.exp(log_sigma) * epsilons) + mean
        
        logits = self.log_softmax(self.dec_projection(sample))
        rec_loss = -1 * torch.sum(logits * input_bows, 1)
        #下面计算KLD
        
        alphas = torch.ones_like(posterior.concentration)
        prior = Dirichlet(alphas)
        
        # kld = F.kl_div(sample, epsilons, size_average=True)
        kld = kl_divergence(posterior, prior).cuda()
        return sample, logits, kld, rec_loss

if __name__ == "__main__":
    # dataset = "mr"
    # BoW = np.load(f"temp/{dataset}.BoW.npy")
    # BoW = BoW[:1000]
    # plsa = PLSA(docs[:100].numpy(), 20, vocab)
    # P_wi_zk, P_zk_dj = plsa.calc()
    # K = 20
    # vocab = list(vocab.iloc[:, 0])
    # for k in range(K):
        # sort_inds = np.argsort(P_wi_zk[k])[::-1]  # 对话题zk条件下的P(wi|zk)的值进行降序排列后取出对应的索引值
        # topic = []  # 定义一个空列表用于保存话题zk概率最大的前10个单词
        # for i in range(10):
            # topic.append(vocab[sort_inds[i]])
        # # topic = ' '.join(topic)  # 将10个单词以空格分隔，构成对话题zk的文本表述
        # print('Topic {}: {}'.format(k + 1, topic))  # 打印话题zk
    x = torch.randn(8,40).cuda()
    lda = ProdLDA(40).cuda()
    sample, logits, kld, rec_loss = lda(x)
    print(sample.shape)
    print(logits.shape)
    print(kld.shape)
    print(rec_loss.shape)