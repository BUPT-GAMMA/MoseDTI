from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
import dgl


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data



class RelationPretrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size  # 256
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))  # 1/sqrt(hybrid freq of the pos triple)

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    # a bool ndarray of len negative_sample. if negative_sample ele not in self.true_head[(relation, tail)], the corresponding position is true.
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode  # negative_sample 256

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode  # negative_sample 1024(bs)*256

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Generate a dict of (head, relation) and (tail, -relation-1)
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail



class TestRelationPretrainDataset(Dataset):
    def __init__(self, triples):
        self.len = len(triples)
        self.triples = triples


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_ for _ in data], dim=0)

        return positive_sample



class SelfDataset(Dataset):
    def __init__(self, pos_triples, neg_triples, ent_id2smiles, ent_id2seq):
        super().__init__()
        self.samples = list()
        for i, tri in enumerate(pos_triples + neg_triples):
            if i < len(pos_triples):
                self.samples.append((tri, 1))
            else:
                self.samples.append((tri, 0))


        self.ent_id2smiles = ent_id2smiles
        self.ent_id2seq = ent_id2seq


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        tri, label = self.samples[idx]
        head, relation, tail = tri
        smiles_feat = self.ent_id2smiles[head]
        seq_feat = self.ent_id2seq[tail]
        return smiles_feat, seq_feat, label, head, tail

    @staticmethod
    def collate_fn(batch):
        smiles_gs = dgl.batch([_[0][0] for _ in batch])
        smiles_hs = list()
        for sample in batch:
            smiles_hs += sample[0][1]
        smiles_hs = torch.tensor(smiles_hs, dtype=torch.float32)

        seq_feats = torch.tensor([_[1] for _ in batch], dtype=torch.int64)
        labels = torch.tensor([_[2] for _ in batch])
        heads = torch.tensor([_[3] for _ in batch])
        tails = torch.tensor([_[4] for _ in batch])
        return (smiles_gs, smiles_hs), seq_feats, labels, heads, tails
