# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from logging import getLogger
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import json

from .utils import reverse_sentences, clip_parameters
from .utils import get_optimizer, parse_lambda_config, update_lambdas
from .model import build_mt_model
from .multiprocessing_event_loop import MultiprocessingEventLoop
from .test import test_sharing


logger = getLogger()


class TrainerMT(MultiprocessingEventLoop):

    VALIDATION_METRICS = []

    def __init__(self, encoder, decoder, discriminator, lm, data, params):
        """
        Initialize trainer.
        """
        super().__init__(device_ids=tuple(range(params.otf_num_processes)))
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.lm = lm
        self.data = data
        self.params = params

        # previous sentence weights
        self.prev_sent_weights = {}
        for lang1 in data['mono']:
            data_size = data['mono'][lang1]['train'].__len__()
            self.prev_sent_weights[lang1] = torch.FloatTensor(data_size).fill_(-1)
            if False: #true
                self.prev_sent_weights[lang1] = np.array([None]*data_size)

        # initialization for on-the-fly generation/training
        #if len(params.pivo_directions) > 0:
        #    self.otf_start_multiprocessing()

        # define encoder parameters (the ones shared with the
        # decoder are optimized by the decoder optimizer)
        enc_params = list(encoder.parameters())
        for i in range(params.n_langs):
            if params.share_lang_emb and i > 0:
                break
            assert enc_params[i].size() == (params.n_words[i], params.emb_dim)
        if self.params.share_encdec_emb:
            to_ignore = 1 if params.share_lang_emb else params.n_langs
            enc_params = enc_params[to_ignore:]

        # optimizers
        if params.dec_optimizer == 'enc_optimizer':
            params.dec_optimizer = params.enc_optimizer
        self.enc_optimizer = get_optimizer(enc_params, params.enc_optimizer) if len(enc_params) > 0 else None
        self.dec_optimizer = get_optimizer(decoder.parameters(), params.dec_optimizer)
        self.dis_optimizer = get_optimizer(discriminator.parameters(), params.dis_optimizer) if discriminator is not None else None
        self.lm_optimizer = get_optimizer(lm.parameters(), params.enc_optimizer) if lm is not None else None

        # models / optimizers
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }
        self.sample_size = 0

        # define validation metrics / stopping criterion used for early stopping
        logger.info("Stopping criterion: %s" % params.stopping_criterion)
        if params.stopping_criterion == '':
            #for lang1, lang2 in self.data['para'].keys():
            for lang1, lang2 in self.params.para_directions:
                for data_type in ['valid', 'test']:
                    self.VALIDATION_METRICS.append('bleu_%s_%s_%s' % (lang1, lang2, data_type))
            for lang1, lang2, lang3 in self.params.pivo_directions:
                if lang1 == lang3:
                    continue
                for data_type in ['valid', 'test']:
                    self.VALIDATION_METRICS.append('bleu_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type))
            self.stopping_criterion = None
            self.best_stopping_criterion = None
        else:
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            self.stopping_criterion = split[0]
            self.best_stopping_criterion = -1e12
            assert len(self.VALIDATION_METRICS) == 0
            self.VALIDATION_METRICS.append(self.stopping_criterion)

        # training variables
        self.best_metrics = {metric: -1e12 for metric in self.VALIDATION_METRICS}
        self.epoch = 0
        self.epoch_num = {}
        self.n_total_iter = 0
        self.freeze_enc_emb = self.params.freeze_enc_emb
        self.freeze_dec_emb = self.params.freeze_dec_emb

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'dis_costs': [],
            'processed_s': 0,
            'processed_w': 0,
        }
        for lang in params.mono_directions:
            self.stats['xe_costs_%s_%s' % (lang, lang)] = []
        for lang1, lang2 in params.para_directions:
            self.stats['xe_costs_%s_%s' % (lang1, lang2)] = []
        for lang1, lang2 in params.back_directions:
            self.stats['xe_costs_bt_%s_%s' % (lang1, lang2)] = []
        for lang1, lang2, lang3 in params.pivo_directions:
            self.stats['xe_costs_%s_%s_%s' % (lang1, lang2, lang3)] = []
        for lang in params.langs:
            self.stats['lme_costs_%s' % lang] = []
            self.stats['lmd_costs_%s' % lang] = []
            self.stats['lmer_costs_%s' % lang] = []
            self.stats['enc_norms_%s' % lang] = []
        self.last_time = time.time()
        if len(params.pivo_directions) > 0:
            self.gen_time = 0

        # data iterators
        self.iterators = {}

        # initialize BPE subwords
        self.init_bpe()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params, 'lambda_xe_mono')
        parse_lambda_config(params, 'lambda_xe_para')
        parse_lambda_config(params, 'lambda_xe_back')
        parse_lambda_config(params, 'lambda_xe_otfd')
        parse_lambda_config(params, 'lambda_xe_otfa')
        parse_lambda_config(params, 'lambda_dis')
        parse_lambda_config(params, 'lambda_lm')

    def init_bpe(self):
        """
        Index BPE words.
        """
        self.bpe_end = []
        for lang in self.params.langs:
            dico = self.data['dico'][lang]
            self.bpe_end.append(np.array([not dico[i].endswith('@@') for i in range(len(dico))]))

    def get_iterator(self, iter_name, lang1, lang2, back, group):
        """
        Create a new iterator for a dataset.
        """
        assert back is False or lang2 is not None
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '') + ('_group' if group else '')
        logger.info("Creating new training %s iterator ..." % key)
        if lang2 is None:
            dataset = self.data['mono'][lang1]['train'] 
        elif group:
            dataset = self.data['group_mono'][(lang1, lang2)]['train']
        elif back:
            dataset = self.data['back'][(lang1, lang2)]
        else:
            k = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            dataset = self.data['para'][k]['train']
        if 'otf' in iter_name and self.params.curri:
            assert self.params.max_tokens != -1
            if not f'{lang1}_{lang2}' in self.epoch_num:
                self.epoch_num[f'{lang1}_{lang2}'] = 0
            cur_epoch = self.epoch_num[f'{lang1}_{lang2}']
            c_0 = 0.1
            #sample_ratio = np.minimum(np.power(cur_epoch/25*(1-c_0**2) + c_0**2, 1/2), 1)
            #sample_ratio = self.epoch/30
            #sample_ratio = cur_epoch/self.params.curri_max_epoch + 0.05
            sample_ratio = np.minimum(np.power(cur_epoch/self.params.curri_max_epoch*(1-c_0**2) + c_0**2, 1/2), 1)
            sample_ratio = np.minimum(sample_ratio, 1)
            print(sample_ratio)
            print(f'cur_epoch {cur_epoch}')
            if not self.params.eval_epoch: 
                self.epoch_num[f'{lang1}_{lang2}'] += 1
            iterator = dataset.get_curri_max_token_iterator(shuffle=self.params.data_shuffle, group_by_size=self.params.group_by_size, iter_name=iter_name, sample_ratio=sample_ratio)
        else:
            if self.params.max_tokens == -1:
                iterator = dataset.get_iterator(shuffle=self.params.data_shuffle, group_by_size=self.params.group_by_size, iter_name=iter_name)()
            else:
                iterator = dataset.get_max_token_iterator(shuffle=self.params.data_shuffle, group_by_size=self.params.group_by_size, iter_name=iter_name)
        self.iterators[key] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2, back=False, group=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert back is False or lang2 is not None
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        key = ','.join([x for x in [iter_name, lang1, lang2] if x is not None]) + ('_back' if back else '') + ('_group' if group else '')
        iterator = self.iterators.get(key, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, back, group)
        try:
            batch = next(iterator)
        except StopIteration: #end of one epoch
            if self.params.eval_epoch:
                if not f'{lang1}_{lang2}' in self.epoch_num:
                    self.epoch_num[f'{lang1}_{lang2}'] = 0
                self.epoch_num[f'{lang1}_{lang2}'] += 1
                if self.epoch_num[f'{lang1}_{lang2}'] % self.params.eval_epoch_num == 0:
                    self.logger.info("====================== End of epoch %i ======================" % self.epoch_num[f'{lang1}_{lang2}'])
                    scores = self.evaluator.run_all_evals(self.epoch)
                    #scores = self.evaluator.run_para_evals(self.epoch_num[f'{lang1}_{lang2}'], lang1, lang2)
                    # print / JSON log
                    for k, v in scores.items():
                        self.logger.info('%s -> %.6f' % (k, v))
                    self.logger.info("__log__:%s" % json.dumps(scores))
                    # save best / save periodic / end epoch
                    self.save_best_model(scores)
                    self.save_periodic()
                    self.save_checkpoint()

            iterator = self.get_iterator(iter_name, lang1, lang2, back, group)
            batch = next(iterator)
        return batch if (lang2 is None or lang1 < lang2 or back) else batch[::-1]

    def zero_grad(self, models):
        """
        Zero gradients.
        """
        if type(models) is not list:
            models = [models]
        models = [self.model_opt[name] for name in models]
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.zero_grad()

    def update_params(self, models, multiply_grad=1):
        """
        Update parameters.
        """
        if type(models) is not list:
            models = [models]
        # don't update encoder when it's frozen
        models = [self.model_opt[name] for name in models]
        # clip gradients
        for model, _ in models:
            if self.params.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), self.params.clip_grad_norm)

        # optimizer
        for _, optimizer in models:
            if optimizer is not None:
                if hasattr(optimizer, 'multiply_grads'):
                    optimizer.multiply_grads(1.0 / float(multiply_grad))
                else:
                    for model, _ in models:
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad *= 1.0 / float(multiply_grad) 
                optimizer.step()

    def get_lrs(self, models):
        """
        Get current optimizer learning rates.
        """
        if type(models) is not list:
            models = [models]
        lrs = {}
        for name in models:
            optimizer = self.model_opt[name][1]
            if optimizer is not None:
                lrs[name] = optimizer.param_groups[0]['lr']
        return lrs

    def enc_dec_step(self, lang1, lang2, lambda_xe, back=False, inner_step=1):
        """
        Source / target autoencoder training (parallel data):
            - encoders / decoders training on cross-entropy
            - encoders training on discriminator feedback
            - encoders training on L2 loss (seq2seq only, not for attention)
        """
        params = self.params
        assert lang1 in params.langs and lang2 in params.langs
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        loss_fn = self.decoder.loss_fn[lang2_id]
        n_words = params.n_words[lang2_id]
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # batch
        if back:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2, back=True)
        elif lang1 == lang2:
            sent1, len1 = self.get_batch('encdec', lang1, None)
            sent2, len2 = sent1, len1
        else:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2)

        # prepare the encoder / decoder inputs
        if lang1 == lang2:
            sent1, len1 = self.add_noise(sent1, len1, lang1_id)
        sent1, sent2 = sent1.cuda(), sent2.cuda()

        # encoded states
        if params.weighted_loss:
            weight = torch.FloatTensor(sent1.size(1)).fill_(1)#.cuda()
            #if params.encoder_weight:
            if False: #weight parallel data
                with torch.no_grad():
                    encoded = self.encoder(sent1, len1, lang1_id)
                    mask1 = encoded.dec_input['encoder_padding_mask'].unsqueeze(2).float().detach()
                    mask1 = 1 - mask1
                    v1 = encoded.dec_input['encoder_out'].transpose(0, 1).detach() #.mean(0),  # T x B x C
                    v1 = v1*mask1
                    v1 = v1.mean(1)
                    encoded2 = self.encoder(sent2.detach(), len2.detach(), lang_id=lang2_id)
                    mask2 = encoded2.dec_input['encoder_padding_mask'].unsqueeze(2).float()
                    mask2 = 1 - mask2
                    v2 = encoded2.dec_input['encoder_out'].transpose(0, 1) #.mean(0),  # T x B x C
                    v2 = v2*mask2
                    v2 = v2.mean(1)
                    weight = F.cosine_similarity(v1, v2)

            #if params.ratio_weight:
            if False: # weight parallel data
                with torch.no_grad():

                    bs = sent1.size(1)

                    n_words2 = params.n_words[lang2_id]
                    encoded2 = self.encoder(sent1, len1, lang1_id)
                    forward_scores = self.decoder(encoded2, sent2[:-1], lang2_id)
                    forward_scores = forward_scores.view(-1, n_words2)
                    forward_scores = F.log_softmax(forward_scores, dim=-1)
                    sent2_ref = sent2[1:].view(-1, 1)
                    ll = forward_scores.gather(dim=-1, index=sent2_ref)
                    ll = ll.view(-1, bs)
                    non_pad_mask = sent2[1:].ne(self.params.pad_index)
                    ll = ll* (non_pad_mask.float())
                    forward_scores = ll.sum(0)
                    assert forward_scores.size() == (bs,)

                    n_words1 = params.n_words[lang1_id]
                    encoded1 = self.encoder(sent2, len2, lang2_id)
                    backward_scores = self.decoder(encoded1, sent1[:-1], lang1_id)
                    backward_scores = backward_scores.view(-1, n_words1)
                    backward_scores = F.log_softmax(backward_scores, dim=-1)
                    sent1_ref = sent1[1:].view(-1, 1)
                    ll = backward_scores.gather(dim=-1, index=sent1_ref)
                    ll = ll.view(-1, bs)
                    non_pad_mask = sent1[1:].ne(self.params.pad_index)
                    ll = ll* (non_pad_mask.float())
                    backward_scores = ll.sum(0)
                    assert backward_scores.size() == (bs,)

                    weight = torch.exp(-abs(backward_scores/len2.float().cuda() - forward_scores/len1.float().cuda()))
                

            encoded = self.encoder(sent1, len1, lang1_id)
            self.stats['enc_norms_%s' % lang1].append(encoded.dis_input.data.norm(2, 1).mean().item())
            # cross-entropy scores / loss
            scores = self.decoder(encoded, sent2[:-1], lang2_id)

            weight = weight.cuda()
            xe_loss, sample_size = loss_fn(scores.view(-1, n_words), sent2[1:].view(-1), weight, sent2.size(0)-1, size_average=False)
        else:
            1/0
            xe_loss, sample_size = loss_fn(scores.view(-1, n_words), sent2[1:].view(-1), size_average=False)
        if back:
            self.stats['xe_costs_bt_%s_%s' % (lang1, lang2)].append(xe_loss.item())
        else:
            self.stats['xe_costs_%s_%s' % (lang1, lang2)].append(xe_loss.item())

        # discriminator feedback loss
        if params.lambda_dis:
            predictions = self.discriminator(encoded.dis_input.view(-1, encoded.dis_input.size(-1)))
            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.sample_size += sample_size
        loss.backward()
        if (inner_step + 1) % self.params.update_freq == 0:
            self.update_params(['enc', 'dec'], multiply_grad=self.sample_size)
            self.zero_grad(['enc', 'dec'])
            self.sample_size = 0

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()


    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        n_batches = len(self.params.mono_directions) + len(self.params.para_directions) + len(self.params.back_directions) + len(self.params.pivo_directions)
        self.n_sentences += n_batches * self.params.batch_size
        self.print_stats()
        update_lambdas(self.params, self.n_total_iter)

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 50 == 0:
            mean_loss = [
                ('DIS', 'dis_costs'),
            ]
            for lang in self.params.mono_directions:
                mean_loss.append(('XE-%s-%s' % (lang, lang), 'xe_costs_%s_%s' % (lang, lang)))
            for lang1, lang2 in self.params.para_directions:
                mean_loss.append(('XE-%s-%s' % (lang1, lang2), 'xe_costs_%s_%s' % (lang1, lang2)))
            for lang1, lang2 in self.params.back_directions:
                mean_loss.append(('XE-BT-%s-%s' % (lang1, lang2), 'xe_costs_bt_%s_%s' % (lang1, lang2)))
            for lang1, lang2, lang3 in self.params.pivo_directions:
                mean_loss.append(('XE-%s-%s-%s' % (lang1, lang2, lang3), 'xe_costs_%s_%s_%s' % (lang1, lang2, lang3)))
            for lang in self.params.langs:
                mean_loss.append(('LME-%s' % lang, 'lme_costs_%s' % lang))
                mean_loss.append(('LMD-%s' % lang, 'lmd_costs_%s' % lang))
                mean_loss.append(('LMER-%s' % lang, 'lmer_costs_%s' % lang))
                mean_loss.append(('ENC-L2-%s' % lang, 'enc_norms_%s' % lang))

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(self.stats['processed_s'] * 1.0 / diff,
                                                                   self.stats['processed_w'] * 1.0 / diff)
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['enc', 'dec'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # generation time
            if len(self.params.pivo_directions) > 0:
                s_time = " - Sentences generation time: % .2fs (%.2f%%)" % (self.gen_time, 100. * self.gen_time / diff)
                self.gen_time = 0
            else:
                s_time = ""

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr + s_time)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({
            'enc': self.encoder,
            'dec': self.decoder,
            'dis': self.discriminator,
            'lm': self.lm,
        }, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        checkpoint_data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'discriminator': self.discriminator,
            'lm': self.lm,
            'enc_optimizer': self.enc_optimizer,
            'dec_optimizer': self.dec_optimizer,
            'dis_optimizer': self.dis_optimizer,
            'lm_optimizer': self.lm_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(checkpoint_data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            return False
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.encoder = checkpoint_data['encoder']
        self.decoder = checkpoint_data['decoder']
        self.discriminator = checkpoint_data['discriminator']
        self.lm = checkpoint_data['lm']
        self.enc_optimizer = checkpoint_data['enc_optimizer']
        self.dec_optimizer = checkpoint_data['dec_optimizer']
        self.dis_optimizer = checkpoint_data['dis_optimizer']
        self.lm_optimizer = checkpoint_data['lm_optimizer']
        self.epoch = checkpoint_data['epoch']
        self.n_total_iter = checkpoint_data['n_total_iter']
        self.best_metrics = checkpoint_data['best_metrics']
        self.best_stopping_criterion = checkpoint_data['best_stopping_criterion']
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)
        return True

    def test_sharing(self):
        """
        Test to check that parameters are shared correctly.
        """
        test_sharing(self.encoder, self.decoder, self.lm, self.params)
        logger.info("Test: Parameters are shared correctly.")

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric in self.VALIDATION_METRICS:
            if not metric in scores:
                print(f'{metric} not in scores!')
                continue
            if scores[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.params.save_periodic and self.epoch % 20 == 0 and self.epoch > 0:
            self.save_model('periodic-%i' % self.epoch)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        if self.params.eval_epoch:
            return
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None:
            assert self.stopping_criterion in scores
            if scores[self.stopping_criterion] > self.best_stopping_criterion:
                self.best_stopping_criterion = scores[self.stopping_criterion]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            if scores[self.stopping_criterion] < self.best_stopping_criterion:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                exit()
        self.epoch += 1
        self.save_checkpoint()

    def register_evaluator(self, evaluator):
        self.evaluator = evaluator

    def register_logger(self, logger):
        self.logger = logger

    def bt_step(self, direction, lambda_xe, backprop_temperature, inner_step=1, batch=None):
        params = self.params

        #print('start')
        lang1, lang2, lang3 = direction
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        # 2-lang back-translation - parallel data
        assert lang1 == lang3 != lang2

        batch = self.get_batch('otf', lang1, None) if batch is None else batch
        sent1, len1, sent_ids = batch

        if lambda_xe == 0:
            logger.warning("Unused generated CPU batch for direction %s-%s-%s!" % (lang1, lang2, lang1))
            return

        sent1 = sent1.cuda()

        self.encoder.eval()
        self.decoder.eval()

        loss = 0

        with torch.no_grad():
            bs = sent1.size(1)
            encoded = self.encoder(sent1, len1, lang_id=lang1_id)
            max_len = int(1.2 * len1.max() + 10)
            if params.otf_sample == -1:
                weight = torch.FloatTensor(bs).fill_(1).cuda()
                sent2, len2, back_scores2 = self.decoder.generate(encoded, lang_id=lang2_id, max_len=max_len, return_score=params.back_weight)
                if params.back_weight:
                    weight = weight * torch.exp(back_scores2)

                if params.forward_weight:
                    encoded2 = self.encoder(sent2, len2, lang_id=lang2_id)
                    forward_scores2 = self.decoder(encoded2, sent1[:-1], lang_id=lang1_id)
                    n_words1 = params.n_words[lang1_id]

                    forward_scores2 = forward_scores2.view(-1, n_words1)
                    forward_scores2 = F.log_softmax(forward_scores2, dim=-1)
                    sent1_ref = sent1[1:].view(-1, 1)
                    
                    ll = forward_scores2.gather(dim=-1, index=sent1_ref)
                    ll = ll.view(-1, bs)
                    non_pad_mask = sent1[1:].ne(self.params.pad_index)
                    ll = ll* (non_pad_mask.float())
                    forward_scores2 = ll.sum(0)
                    #forward_scores2 = ll.mean(0)
                    assert forward_scores2.size() == (bs,)

                    weight = weight * torch.exp(forward_scores2)
                
                if params.ratio_weight:
                    weight = torch.exp(-abs(back_scores2/len2.float().cuda() - forward_scores2/len1.float().cuda()))
                
                if params.encoder_weight:
                    with torch.no_grad():
                        mask1 = encoded.dec_input['encoder_padding_mask'].unsqueeze(2).float()
                        mask1 = 1 - mask1
                        v1 = encoded.dec_input['encoder_out'].transpose(0, 1) #.mean(0),  # T x B x C
                        v1 = v1*mask1
                        v1 = v1.mean(1)
                        #print(v1.size())
                        encoded2 = self.encoder(sent2, len2, lang_id=lang2_id)
                        mask2 = encoded2.dec_input['encoder_padding_mask'].unsqueeze(2).float()
                        mask2 = 1 - mask2
                        v2 = encoded2.dec_input['encoder_out'].transpose(0, 1) #.mean(0),  # T x B x C
                        v2 = v2*mask2
                        v2 = v2.mean(1)
                        weight = F.cosine_similarity(v1, v2)

            else:
                assert params.otf_sample_num > 0
                weights = []
                sent2s = []
                len2s = []
                total_weight = torch.FloatTensor(bs).fill_(0).cuda()
                for _ in range(params.otf_sample_num):
                    weight = torch.FloatTensor(bs).fill_(0).cuda()
                    sent2, len2, back_scores2 = self.decoder.generate(encoded, lang_id=lang2_id, max_len=max_len,
                                                           sample=True, temperature=params.otf_sample, return_score=params.back_weight)

                    if params.back_weight:
                        weight = weight + (back_scores2)

                    if params.forward_weight:
                        encoded2 = self.encoder(sent2, len2, lang_id=lang2_id)
                        forward_scores2 = self.decoder(encoded2, sent1[:-1], lang_id=lang1_id)
                        n_words1 = params.n_words[lang1_id]

                        forward_scores2 = forward_scores2.view(-1, n_words1)
                        forward_scores2 = F.log_softmax(forward_scores2, dim=-1)
                        sent1_ref = sent1[1:].view(-1, 1)
                        
                        ll = forward_scores2.gather(dim=-1, index=sent1_ref)
                        ll = ll.view(-1, bs)
                        non_pad_mask = sent1[1:].ne(self.params.pad_index)
                        ll = ll* (non_pad_mask.float())
                        forward_scores2 = ll.mean(0)
                        assert forward_scores2.size() == (bs,)

                        weight = weight + forward_scores2

                    weight = torch.exp(weight)
                    total_weight = total_weight + weight

                    weights.append(weight)
                    sent2s.append(sent2)
                    len2s.append(len2)
                        
                if params.norm_weight:
                    for sent2, len2, weight in zip(sent2s, len2s, weights):
                            weight =  (weight/(total_weight+1e-6))

            assert all(x.is_cuda for x in [sent1, sent2])

        #back translation
        direction = (lang1, lang2, lang1)
        assert direction in params.pivo_directions
        loss_fn = self.decoder.loss_fn[lang1_id]
        n_words2 = params.n_words[lang2_id]
        n_words1 = params.n_words[lang1_id]
        self.encoder.train()
        self.decoder.train()

        if params.forward_loss:
            loss_fnf = self.decoder.loss_fn[lang2_id]
            scores_f = self.decoder(encoded, sent2[:-1], lang_id=lang2_id)
            if params.weighted_loss:
                weightf = weight.cuda()
                xe_loss, sample_size = loss_fnf(scores_f.view(-1, n_words2), sent2[1:].view(-1), weightf, sent2.size(0)-1, size_average=False)
            else:
                xe_loss, sample_size = loss_fn(scores_f.view(-1, n_words2), sent2[1:].view(-1), size_average=False)
            loss = lambda_xe * xe_loss
            # check NaN
            if (loss != loss).data.any():
                logger.error("NaN detected")
                exit()
            self.sample_size += sample_size
            loss.backward()

        if params.backward_loss:
            if params.weighted_loss:
                if params.diff_weight:
                    from copy import deepcopy
                    prev_sent_weights_copy = deepcopy(self.prev_sent_weights[lang1][sent_ids]).cuda()
                    self.prev_sent_weights[lang1][sent_ids] = weight.cpu()
                    one_mask = prev_sent_weights_copy.eq(-1)
                    diff_weight = weight / (prev_sent_weights_copy + 1e-9)
                    diff_weight = torch.clamp(diff_weight, 0.5, 2)
                    #weight = torch.clamp(weight, 0.33, 3)
                    diff_weight[one_mask] = 1
                    weight = weight*diff_weight

                    if False: #Use current models to evaluate sentence quality in the previous epoch
                        prev_sent_weights_copy = deepcopy(self.prev_sent_weights[lang1][sent_ids])
                        self.prev_sent_weights[lang1][sent_ids] = (list(zip(sent2.detach().cpu().transpose(0, 1), len2.detach().cpu())))
                        #self.prev_sent_weights[lang1][sent_ids] = np.array(list(zip(sent2.cpu(), len2.cpu())))
                        not_none = np.where(prev_sent_weights_copy!=None)[0]
                        none = np.where(prev_sent_weights_copy==None)[0]
                        diff_weight = torch.ones(len(sent_ids))
                        if len(not_none) > 0:
                            with torch.no_grad():
                                old_sent2, old_len2 = list(zip(*prev_sent_weights_copy[not_none]))
                                old_len2 = torch.LongTensor(old_len2)
                                old_sent22 = torch.LongTensor(old_len2.size(0), old_len2.max()).fill_(2)
                                for i, s in enumerate(old_sent2):
                                    #print(old_sent2[i][:old_len2[i]])
                                    old_sent22[i][:old_len2[i]].copy_(old_sent2[i][:old_len2[i]].clone())
                                old_sent2 = old_sent22.cuda().transpose(0, 1)

                                if params.encoder_weight:
                                    old_encoded2 = self.encoder(old_sent2, old_len2, lang_id=lang2_id)
                                    old_mask2 = old_encoded2.dec_input['encoder_padding_mask'].unsqueeze(2).float()
                                    old_mask2 = 1 - old_mask2
                                    old_v2 = old_encoded2.dec_input['encoder_out'].transpose(0, 1) #.mean(0),  # T x B x C
                                    old_v2 = old_v2*old_mask2
                                    old_v2 = old_v2.mean(1)
                                    old_weight = F.cosine_similarity(v1, old_v2).cpu().detach()
                                else:
                                    encoded1 = self.encoder(sent1, len1, lang_id=lang1_id)
                                    backward_scores2 = self.decoder(encoded1, old_sent2[:-1], lang_id=lang2_id)
                                    n_words2 = params.n_words[lang2_id]

                                    backward_scores2 = backward_scores2.view(-1, n_words2)
                                    backward_scores2 = F.log_softmax(backward_scores2, dim=-1)
                                    old_sent2_ref = old_sent2[1:].contiguous().view(-1, 1)
                                    
                                    ll = backward_scores2.gather(dim=-1, index=old_sent2_ref)
                                    ll = ll.view(-1, bs)
                                    non_pad_mask = old_sent2[1:].ne(self.params.pad_index)
                                    ll = ll* (non_pad_mask.float())
                                    backward_scores2 = ll.sum(0)
                                    assert backward_scores2.size() == (bs,)

                                    old_encoded2 = self.encoder(old_sent2, old_len2, lang_id=lang2_id)
                                    forward_scores2 = self.decoder(old_encoded2, sent1[:-1], lang_id=lang1_id)
                                    n_words1 = params.n_words[lang1_id]

                                    forward_scores2 = forward_scores2.view(-1, n_words1)
                                    forward_scores2 = F.log_softmax(forward_scores2, dim=-1)
                                    sent1_ref = sent1[1:].view(-1, 1)
                                    
                                    ll = forward_scores2.gather(dim=-1, index=sent1_ref)
                                    ll = ll.view(-1, bs)
                                    non_pad_mask = sent1[1:].ne(self.params.pad_index)
                                    ll = ll* (non_pad_mask.float())
                                    forward_scores2 = ll.sum(0)
                                    assert forward_scores2.size() == (bs,)

                                    #weight = weight * torch.exp(forward_scores2)
                                    old_weight = torch.exp(-abs(backward_scores2.cpu()/old_len2.float() - forward_scores2.cpu()/len1.float()))
                                    
                                #diff_weight = np.ones(len(sent_ids))
                                diff_weight[not_none] = (old_weight+1e-9).clone()
                                diff_weight = weight/diff_weight.cuda()
                                #diff_weight = weight.cpu().numpy()/diff_weight
                                diff_weight = torch.clamp(diff_weight, 0.5, 2).detach()
                                weight = weight*diff_weight
                        weight = diff_weight
                    
                    
                if params.random_weight:
                    weight = torch.FloatTensor(weight.size()).uniform_(0.5, 2.0)
                #weight += (1-weight.mean())
                weight = weight.detach()
                weight = weight.cuda()
                #xe_loss, sample_size = loss_fn(scores.view(-1, n_words1), sent1[1:].view(-1), weight, sent1.size(0)-1, size_average=False)
            else:
                1/0
                xe_loss, sample_size = loss_fn(scores.view(-1, n_words1), sent1[1:].view(-1), size_average=False)

            encoded = self.encoder(sent2, len2, lang_id=lang2_id)
            scores = self.decoder(encoded, sent1[:-1], lang_id=lang1_id)
            xe_loss, sample_size = loss_fn(scores.view(-1, n_words1), sent1[1:].view(-1), weight, sent1.size(0)-1, size_average=False)

            assert lambda_xe > 0
            loss = lambda_xe * xe_loss
            # check NaN
            if (loss != loss).data.any():
                logger.error("NaN detected")
                exit()
            # optimizer
            self.sample_size += sample_size
            loss.backward()
        self.stats['xe_costs_%s_%s_%s' % direction].append(xe_loss.item())
        if (inner_step + 1) % self.params.update_freq == 0:
            assert params.otf_update_enc or params.otf_update_dec
            to_update = []
            if params.otf_update_enc:
                to_update.append('enc')
            if params.otf_update_dec:
                to_update.append('dec')
            self.update_params(to_update, multiply_grad=self.sample_size)
            self.zero_grad(to_update)
            self.sample_size = 0

        # number of processed sentences / words
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += len1.sum()
        #print('end')

