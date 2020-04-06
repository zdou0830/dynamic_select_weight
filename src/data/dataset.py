# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import os
import numpy as np
import torch
import itertools


logger = getLogger()


class Dataset(object):

    def __init__(self, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.unk_index = params.unk_index
        self.bos_index = params.bos_index
        self.batch_size = params.batch_size
        self.max_tokens = params.max_tokens

    def batch_sentences(self, sentences, lang_id, sentence_ids=None, left_pad=False):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (s_len, n) where s_len is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        if sentences is None:
            return None
        assert type(lang_id) is int
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max(), lengths.size(0)).fill_(self.pad_index)

        if not left_pad:
            sent[0] = self.bos_index[lang_id]
            for i, s in enumerate(sentences):
                sent[1:lengths[i] - 1, i].copy_(s)
                sent[lengths[i] - 1, i] = self.eos_index
        else:
            for i, s in enumerate(sentences):
                sent[-lengths[i], i] = self.bos_index[lang_id]
                sent[-lengths[i] + 1:-1, i].copy_(s)
                sent[-1, i] = self.eos_index

        if sentence_ids is None:
            return sent, lengths
        else:
            return sent, lengths, sentence_ids


class MonolingualDataset(Dataset):

    def __init__(self, sent, pos, dico, lang_id, params, mono_weight_file="", diff_weight_file="", info_weight_file=""):
        super(MonolingualDataset, self).__init__(params)
        assert type(lang_id) is int
        self.sent = sent
        self.pos = pos
        self.dico = dico
        self.lang_id = lang_id
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.is_parallel = False
        self.left_pad = params.left_pad
        self.mono_weight = self.load_mono_weight(mono_weight_file) if not mono_weight_file == '' else np.zeros([len(self.pos)])
        self.use_diff_info = False
        if not diff_weight_file == '':
            self.use_diff_info = True
            self.diff_weight = self.load_mono_weight(diff_weight_file)
        else:
            self.diff_weight = np.zeros([len(self.pos)])
        if not info_weight_file == '':
            self.info_weight = self.load_mono_weight(info_weight_file)
        else:
            self.info_weight = np.zeros([len(self.pos)])
        self.max_len=self.lengths.max()

        # check number of sentences
        assert len(self.pos) == (self.sent == -1).sum()

        self.remove_empty_sentences()

        assert len(pos) == (sent[torch.from_numpy(pos[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent.min() < sent.max() < len(dico)                    # check dictionary indices
        assert self.lengths.min() > 0                                       # check empty sentences

    def load_mono_weight(self, mono_weight_file):
        assert os.path.isfile(mono_weight_file)
        f = open(mono_weight_file, 'r', encoding='utf-8')
        mono_weight = []
        for i, line in enumerate(f):
            try:
                mono_weight.append(float(line))
            except:
                print(line)
                raise ValueError('Weights should be float!')
        a = np.array(mono_weight) 
        return ( a - np.mean(a) ) / np.std(a) #(np.array(mono_weight) 
        return np.array(mono_weight)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.mono_weight = self.mono_weight[indices]
        self.diff_weight = self.diff_weight[indices]
        self.info_weight = self.info_weight[indices]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.mono_weight = self.mono_weight[indices]
        self.diff_weight = self.diff_weight[indices]
        self.info_weight = self.info_weight[indices]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos)
        if a < b:
            self.pos = self.pos[a:b]
            self.lengths = self.pos[:, 1] - self.pos[:, 0]
            self.mono_weight = self.mono_weight[a:b]
            self.diff_weight = self.diff_weight[a:b]
            self.info_weight = self.info_weight[a:b]
        else:
            self.pos = torch.LongTensor()
            self.lengths = torch.LongTensor()
            self.mono_weight = torch.DoubleTensor()
            self.diff_weight = torch.DoubleTensor()
            self.info_weight = torch.DoubleTensor()

    def get_batches_iterator(self, batches, iter_name=None):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos = self.pos[sentence_ids]
                sent = [self.sent[a:b] for a, b in pos]
                yield self.batch_sentences(sent, self.lang_id)

        def iterator_otf():
            for sentence_ids in batches:
                pos = self.pos[sentence_ids]
                sent = [self.sent[a:b] for a, b in pos]
                yield self.batch_sentences(sent, self.lang_id, sentence_ids)
        if 'otf' in iter_name:
            return iterator_otf
        else:
            return iterator

    def gaussian(self, x, sig):
        return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

    def get_mono_prob(self, sample_ratio, var=0.2):
        return sample_ratio*(self.info_weight) + (1-sample_ratio)*(self.diff_weight) + np.random.randn(len(self.pos))*0.02

    def get_rand_prob(self, sample_ratio, mono_prob, var=0.2):
        return np.percentile(mono_prob, 70)

    def weighted_sample(self, mono_prob):
        size = len(mono_prob)
        vec = np.arange(size)
        denom = np.sum(mono_prob)
        mono_prob = np.array(mono_prob)/denom
        return np.random.choice(vec, size=int(0.7*size), replace=False, p = mono_prob)

    def get_group_thres(self, sample_ratio, n_group=10):
        thres = 1.0/n_group
        group_index = int(sample_ratio/thres)
        group_index = min(group_index, n_group-1)
        low_thres = thres*group_index
        high_thres = low_thres + thres
        low_filter = (low_thres <= self.mono_weight)
        high_filter = (high_thres >= self.mono_weight)
        filter_pos_indices = (low_filter & high_filter)
        return filter_pos_indices


    def get_curri_max_token_iterator(self, shuffle, group_by_size=True, n_sentences=-1, iter_name=None, sample_ratio=0):
        """
        Return a sentences iterator.
        """
        mono_prob = self.get_mono_prob(sample_ratio)
        rand_prob = self.get_rand_prob(sample_ratio, mono_prob)
        filter_pos_indices = (mono_prob >= rand_prob)
        filter_pos = self.pos[filter_pos_indices]
        if len(filter_pos) == 0:
            filter_pos_indices[0] = True
            filter_pos = self.pos[filter_pos_indices]
        filter_lengths = filter_pos[:, 1] - filter_pos[:, 0]

        n_sentences = len(filter_pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(filter_pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert 'otf' in iter_name

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(filter_pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(filter_lengths[indices], kind='mergesort')]

        batches = []
        batch = []
        sample_len = 0
        for idx in indices:
            sample_len = max(sample_len, filter_lengths[idx] + 2)
            assert sample_len <= self.max_tokens, "sentence exceeds max_tokens limit!"
            num_tokens = (len(batch) + 1) * sample_len
            if num_tokens > self.max_tokens:
                batches.append(batch)
                batch = []
                sample_len = filter_lengths[idx] + 2
            batch.append(idx)

        if len(batch) > 0:
            batches.append(batch)

        if shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            pos = filter_pos[batch]
            sent = [self.sent[a:b] for a, b in pos]
            if 'otf' in iter_name:
                yield self.batch_sentences(sent, self.lang_id, batch, left_pad=self.left_pad)
            else:
                yield self.batch_sentences(sent, self.lang_id, left_pad=self.left_pad)
            


    def get_max_token_iterator(self, shuffle, group_by_size=True, n_sentences=-1, iter_name=None):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert 'otf' in iter_name

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        batches = []
        batch = []
        sample_len = 0
        for idx in indices:
            sample_len = max(sample_len, self.lengths[idx] + 2)
            assert sample_len <= self.max_tokens, "sentence exceeds max_tokens limit!"
            num_tokens = (len(batch) + 1) * sample_len
            if num_tokens > self.max_tokens:
                batches.append(batch)
                batch = []
                sample_len = self.lengths[idx] + 2
            batch.append(idx)

        if len(batch) > 0:
            batches.append(batch)

        if shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            pos = self.pos[batch]
            sent = [self.sent[a:b] for a, b in pos]
            if 'otf' in iter_name:
                yield self.batch_sentences(sent, self.lang_id, batch, left_pad=self.left_pad)
            else:
                yield self.batch_sentences(sent, self.lang_id, left_pad=self.left_pad)
            


    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, iter_name=None):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches, iter_name=iter_name)


class ParallelDataset(Dataset):

    def __init__(self, sent1, pos1, dico1, lang1_id, sent2, pos2, dico2, lang2_id, params):
        super(ParallelDataset, self).__init__(params)
        assert type(lang1_id) is int
        assert type(lang2_id) is int
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.dico1 = dico1
        self.dico2 = dico2
        self.lang1_id = lang1_id
        self.lang2_id = lang2_id
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.is_parallel = True
        self.src_left_pad = params.src_left_pad
        self.tgt_left_pad = params.tgt_left_pad

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == -1).sum()
        assert len(self.pos2) == (self.sent2 == -1).sum()

        self.remove_empty_sentences()

        print(len(pos1))
        print(len(pos2))
        assert len(pos1) == len(pos2) > 0                                      # check number of sentences
        assert len(pos1) == (sent1[torch.from_numpy(pos1[:, 1])] == -1).sum()  # check sentences indices
        assert len(pos2) == (sent2[torch.from_numpy(pos2[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent1.min() < sent1.max() < len(dico1)                    # check dictionary indices
        assert -1 <= sent2.min() < sent2.max() < len(dico2)                    # check dictionary indices
        assert self.lengths1.min() > 0                                         # check empty sentences
        assert self.lengths2.min() > 0                                         # check empty sentences

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos1)
        if a < b:
            self.pos1 = self.pos1[a:b]
            self.pos2 = self.pos2[a:b]
            self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
            self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        else:
            self.pos1 = torch.LongTensor()
            self.pos2 = torch.LongTensor()
            self.lengths1 = torch.LongTensor()
            self.lengths2 = torch.LongTensor()

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos1 = self.pos1[sentence_ids]
                pos2 = self.pos2[sentence_ids]
                sent1 = [self.sent1[a:b] for a, b in pos1]
                sent2 = [self.sent2[a:b] for a, b in pos2]
                yield self.batch_sentences(sent1, self.lang1_id, left_pad=self.src_left_pad), self.batch_sentences(sent2, self.lang2_id, left_pad=self.tgt_left_pad)
        return iterator

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, iter_name=None):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths2[indices], kind='mergesort')]
            indices = indices[np.argsort(self.lengths1[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)

    def get_max_token_iterator(self, shuffle, group_by_size=False, n_sentences=-1, iter_name=None):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths2[indices], kind='mergesort')]
            indices = indices[np.argsort(self.lengths1[indices], kind='mergesort')]

        batches = []
        batch = []
        sample_len = 0
        for idx in indices:
            cur_len = max(self.lengths1[idx], self.lengths2[idx]) + 2
            sample_len = max(sample_len, cur_len)
            assert sample_len <= self.max_tokens, "sentence exceeds max_tokens limit!"
            num_tokens = (len(batch) + 1) * sample_len
            if num_tokens > self.max_tokens:
                batches.append(batch)
                batch = []
                sample_len = cur_len
            batch.append(idx)

        if len(batch) > 0:
            batches.append(batch)
        if shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            pos1 = self.pos1[batch]
            sent1 = [self.sent1[a:b] for a, b in pos1]
            pos2 = self.pos2[batch]
            sent2 = [self.sent2[a:b] for a, b in pos2]
            yield self.batch_sentences(sent1, self.lang1_id, left_pad=self.src_left_pad), self.batch_sentences(sent2, self.lang2_id, left_pad=self.tgt_left_pad)
            
class GroupMonolingualDataset(Dataset):

    def __init__(self, sent1, pos1, dico1, lang1_id, sent2, pos2, dico2, lang2_id, params, mono_group_file1, mono_group_file2, mono_weight_file1="", mono_weight_file2=""):
        super(GroupMonolingualDataset, self).__init__(params)
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.dico1 = dico1
        self.dico2 = dico2
        self.lang1_id = lang1_id
        self.lang2_id = lang2_id
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.is_parallel = False
        self.left_pad = False
        from collections import defaultdict
        self.group2id = defaultdict(lambda: len(self.group2id))
        self.mono_group1 = self.load_mono_group(mono_group_file1) #if not mono_group_file1 == '' else None #np.ones([len(self.pos)])
        self.mono_group2 = self.load_mono_group(mono_group_file2) #if not mono_group_file2 == '' else None #np.ones([len(self.pos)])
        self.mono_weight1 = self.load_mono_weight(mono_weight_file1) if not mono_weight_file1 == '' else np.ones([len(self.pos1)])
        self.mono_weight2 = self.load_mono_weight(mono_weight_file2) if not mono_weight_file2 == '' else np.ones([len(self.pos2)])

        self.remove_empty_sentences()
        self.remove_long_sentences(params.max_len)

        self.num_cluster = len(self.group2id)
        self.mono_group_ids = [ [[] for i in range(self.num_cluster)] , [[] for i in range(self.num_cluster)] ]
        self.get_mono_group_ids(self.mono_group1, 0)
        self.get_mono_group_ids(self.mono_group2, 1) 

    def get_mono_group_ids(self, mono_group, lang_id):
        for i, group_id in enumerate(mono_group):
            self.mono_group_ids[lang_id][group_id].append(i)

    def load_mono_group(self, mono_group_file):
        assert os.path.isfile(mono_group_file)
        f = open(mono_group_file, 'r', encoding='utf-8')
        mono_group = []
        for i, line in enumerate(f):
            try:
                #group_id = int(line)
                #mono_group.append(group_id)
                group_id = self.group2id[int(line)]
                mono_group.append(group_id)
            except:
                print(line)
                raise ValueError('Group should be int!')
        return np.array(mono_group)

    def load_mono_weight(self, mono_weight_file):
        assert os.path.isfile(mono_weight_file)
        f = open(mono_weight_file, 'r', encoding='utf-8')
        mono_weight = []
        for i, line in enumerate(f):
            try:
                mono_weight.append(float(line))
            except:
                print(line)
                raise ValueError('Weights should be float!')
        return np.array(mono_weight)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.mono_group1 = self.mono_group1[indices]
        self.mono_weight1 = self.mono_weight1[indices]

        indices = np.arange(len(self.pos2))
        indices = indices[self.lengths2[indices] > 0]
        self.pos2 = self.pos2[indices]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.mono_group2 = self.mono_group2[indices]
        self.mono_weight2 = self.mono_weight2[indices]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.mono_group1 = self.mono_group1[indices]
        self.mono_weight1 = self.mono_weight1[indices]

        indices = np.arange(len(self.pos2))
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos2 = self.pos2[indices]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.mono_group2 = self.mono_group2[indices]
        self.mono_weight2 = self.mono_weight2[indices]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos1)
        if a < b:
            self.pos1 = self.pos1[a:b]
            self.pos2 = self.pos2[a:b]
            self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
            self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
            self.mono_group1 = self.mono_group1[a:b]
            self.mono_group2 = self.mono_group2[a:b]
            self.mono_weight1 = self.mono_weight1[a:b]
            self.mono_weight2 = self.mono_weight2[a:b]
        else:
            self.pos1 = torch.LongTensor()
            self.pos2 = torch.LongTensor()
            self.lengths1 = torch.LongTensor()
            self.lengths2 = torch.LongTensor()
            self.mono_group1 = torch.LongTensor() 
            self.mono_group2 = torch.LongTensor()
            self.mono_weight1 = torch.FloatTensor()
            self.mono_weight2 = torch.FloatTensor()

    def get_max_token_iterator(self, shuffle, group_by_size=False, n_sentences=-1, iter_name=None):
        """
        Return a sentences iterator.
        """
        # first
        assert type(shuffle) is bool and type(group_by_size) is bool

        batches = []
        indices = []
        lengths = []

        for cluster_id in range(self.num_cluster): 
            indices1 = self.mono_group_ids[0][cluster_id]
            if indices1 is None:
                indices1 = []
            if len(indices1) > 1:
                # select sentences to iterate over
                if shuffle:
                   np.random.shuffle(indices1)

            indices2 = self.mono_group_ids[1][cluster_id]
            if indices2 is None:
                indices2 = []
            if len(indices2) > 1:
                # select sentences to iterate over
                if shuffle:
                    np.random.shuffle(indices2)
            
            for ind1, ind2 in itertools.zip_longest(indices1, indices2):
                length = 0
                length = length + self.lengths1[ind1] if ind1 is not None else length
                length = length + self.lengths2[ind2] if ind2 is not None else length
                indices.append( (ind1, ind2) )
                lengths.append(length)
                
        #print(self.lengths1[:10].shape)
        #print(lengths[:10].reshape(1))
        if group_by_size:
            indices = np.array(indices)
            indices = indices[np.argsort(lengths, kind='mergesort')]
    
        batches1 = []
        batches2 = []
        batch1 = []
        batch2 = []
        sample_len1 = 0
        sample_len2 = 0
        for idx1, idx2 in indices:
            len1 = self.lengths1[idx1] if not idx1 is None else 0
            len2 = self.lengths2[idx2] if not idx2 is None else 0
            cur_len1 = len1 + 2
            cur_len2 = len2 + 2
            sample_len1 = max(sample_len1, cur_len1)
            sample_len2 = max(sample_len2, cur_len2)
            assert sample_len1 <= self.max_tokens, "sentence exceeds max_tokens limit!"
            assert sample_len2 <= self.max_tokens, "sentence exceeds max_tokens limit!"
            num_tokens1 = (len(batch1) + 1) * sample_len1
            num_tokens2 = (len(batch2) + 1) * sample_len2
            num_tokens = max(num_tokens1, num_tokens2)
            if num_tokens > self.max_tokens:
                batches1.append(batch1)
                batches2.append(batch2)
                batch1 = []
                batch2 = []
                sample_len1 = cur_len1
                sample_len2 = cur_len2
            if not idx1 is None:
                batch1.append(idx1)
            if not idx2 is None:
                batch2.append(idx2)

        if len(batch1) > 0:
            batches1.append(batch1)
        if len(batch2) > 0:
            batches2.append(batch2)

        if shuffle:
            inds = np.random.permutation(len(batches1))
            batches1 = np.array(batches1)
            batches1 = batches1[inds]
            inds = np.random.permutation(len(batches2))
            batches2 = np.array(batches2)
            batches2 = batches2[inds]


        for batch1, batch2 in zip(batches1, batches2):
            sent1 = None
            if len(batch1) > 0:
                pos1 = self.pos1[batch1]
                sent1 = [self.sent1[a:b] for a, b in pos1]
            sent2 = None
            if len(batch2) > 0:
                pos2 = self.pos2[batch2]
                sent2 = [self.sent2[a:b] for a, b in pos2]
            if 'otf' in iter_name:
                yield self.batch_sentences(sent1, self.lang1_id, batch1, left_pad=self.left_pad), self.batch_sentences(sent2, self.lang2_id, batch2, left_pad=self.left_pad)
            else:
                yield self.batch_sentences(sent1, self.lang1_id, left_pad=self.left_pad), self.batch_sentences(sent2, self.lang2_id, left_pad=self.left_pad)
