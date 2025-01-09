"""
a forward of model (assuming batch size is 1) can compute the score for a positive triplet, or scores of all related
negative triplets(certainly containing the positive triplet itself)
There is no use of train_neg.txt
derive from my previous file kge_self.py
"""

# !/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gc

import datetime
import sys

import subprocess

import pickle

from torch.utils.data import DataLoader


from utils_mine import *
from esm_model import *
from few_dataloader import *
import itertools
import random
import time




def main(args, start_time):
    def auto_select_device(use_gpu, gpu_id, remap, required_mem_min=2000, strategy='random'):
        """
        Auto select GPU device
        memory_max: gpu whose used memory exceeding memory_max will no be random selected
        required_mem_min: min required memory of the program
        """

        def get_gpu_memory_map(remap):
            """Get the current gpu usage."""
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,nounits,noheader'
                ], encoding='utf-8')
            gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
            # nvidia GPU id needs to be remapped to align with the cuda id
            # remap = [2, 3, 7, 8, 0, 1, 4, 5, 6, 9]
            return gpu_memory[remap]

        def get_total_gpu_memory_map(remap):
            """Get the total gpu memory."""
            result = subprocess.check_output(
                [
                    'nvidia-smi', '--query-gpu=memory.total',
                    '--format=csv,nounits,noheader'
                ], encoding='utf-8')
            gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
            # nvidia GPU id needs to be remapped to align with the cuda id
            # remap = [2, 3, 7, 8, 0, 1, 4, 5, 6, 9]
            return gpu_memory[remap]

        if torch.cuda.is_available() and use_gpu:
            if gpu_id == -1:
                total_memory_raw = get_total_gpu_memory_map(remap)
                memory_raw = get_gpu_memory_map(remap)
                available_memory = total_memory_raw - memory_raw
                available_memory[0] = available_memory[0] - 24000  # the first gpu is not suggested to use
                # set cuda ids which are not available
                unavailable_gpu = []
                for i, m in enumerate(available_memory):
                    if m < required_mem_min:
                        unavailable_gpu.append(i)
                print('Total GPU Mem: {}'.format(total_memory_raw))
                print('Available GPU Mem: {}'.format(available_memory))
                print('Unselectable GPU ID: {}'.format(unavailable_gpu))
                if strategy == 'random':
                    memory = available_memory / available_memory.sum()
                    memory[unavailable_gpu] = 0
                    gpu_prob = memory / memory.sum()
                    cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                    print('GPU Prob: {}'.format(gpu_prob.round(2)))
                    print(
                        'Random select GPU, select GPU {} with mem: {}'.format(
                            cuda, available_memory[cuda]))
                elif strategy == "max":
                    available_memory[unavailable_gpu] = 0
                    cuda = np.argmax(available_memory)
                    print(
                        'Max select GPU, select GPU {} with mem: {}'.format(
                            cuda, available_memory[cuda]))
                else:
                    raise AssertionError
                return torch.device('cuda:{}'.format(cuda))
            elif 0 <= gpu_id <= 9:
                print(
                    'Manually select GPU, select GPU {}'.format(
                        gpu_id))
                return torch.device('cuda:{}'.format(gpu_id))
            else:
                raise ValueError("The GPU id is invalid！")
        else:
            print('cuda not available')
            return torch.device("cpu")

    class EarlyStopping:
        """Early stops the training if validation loss doesn't improve after a given patience."""

        def __init__(self, directory, save, save_thre, save_mul, fun, patience=15,
                     ):
            """

            """
            self.patience = patience  #15
            self.counter = 0
            self.best_score = float('-inf')
            self.early_stop = False
            self.directory = directory
            self.best_model_path = None
            self.trace_fun = print
            self.save = save
            self.save_thre = save_thre
            self.save_mul = save_mul
            self.fun = fun

        def __call__(self, score, model):
            if (1 - score) >= (1 - self.best_score) * 0.97:
                if_opt = False
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.trace_fun("Training early stopped!")
            else:
                if_opt = True
                self.best_score = score
                self.counter = 0
                self.trace_fun("*****************new opt*******************")
                if self.save and score>self.save_thre:
                    if self.save_mul:
                        save2file(model, self.directory, start_time, info=self.fun + "_" + str(score))
                    else:
                        save2file(model, self.directory, start_time, info=self.fun)
                    self.trace_fun("model saved")


            return self.early_stop, if_opt



    def train_loop(kge_model, optimizer, warm_up_steps, train_iterator, train_step_fun, if_valid=False,
                   valid_dataloader=None, valid_labels=None, test_loop=None, test_module_name=None, test_dataloader=None, test_labels=None):
        #  valid_step save_thre patience
        current_learning_rate = args.learning_rate
        opt_aupr = float("-inf")
        early_stop_count = 0
        model_path = str()
        try:
            for step in range(args.max_relation_steps):  # TODO
                # loss = train_step_fun(kge_model, optimizer, train_iterator, args)
                # if step >= warm_up_steps:
                #     current_learning_rate = current_learning_rate / 10
                #     optimizer = torch.optim.Adam(
                #         filter(lambda p: p.requires_grad, kge_model.parameters()),
                #         lr=current_learning_rate,
                #         weight_decay=args.weight_decay
                #     )
                #     warm_up_steps = warm_up_steps * 3

                if if_valid:  # save according to valid
                    # if (step + 1) % (args.relation_valid_steps // 8) == 0:
                    #     current_time = str(datetime.datetime.now())[0:-4].replace(' ', '_').replace(":", "_")
                    #
                    #     print(f"{current_time} Loss of step {step}: {loss:.8f}")
                    # if (step + 1) % valid_steps == 0:   # compare test performance of the whole test set every a number of steps.
                    if (step + 1) % (args.relation_valid_steps // 1) == 0: # // 8
                        if test_module_name == "relation":
                            _, acc, auc, aupr = test_loop(args, kge_model, valid_dataloader,
                                                          valid_labels)
                        else:
                            raise ValueError


                        print(f"ACC: {acc:.5f}  AUC: {auc:.5f}  AUPR: {aupr:.5f}")

                        if (1 - aupr) < (1 - opt_aupr) * 0.97:
                            opt_aupr = aupr
                            early_stop_count = 0
                            print("***************new opt***************")
                            if 0 < aupr and args.save:  # TODO
                                save2file(kge_model, f"{args.model_path}/", start_time, info="kge")
                                print("Model saved!")
                            _, test_acc, test_auc, test_aupr = test_loop(args, kge_model, test_dataloader,
                                                                         test_labels)
                            print(f"test ACC: {test_acc:.5f} test AUC: {test_auc:.5f}  AUPR: {test_aupr:.5f}#")

                        else:
                            early_stop_count += 1
                            if early_stop_count >= 2:  # 8 6 30 3000
                                print("Early stopped!")
                                return kge_model, test_acc, test_auc, test_aupr



        except KeyboardInterrupt:
            pass



    def load_model_to_device(model_path, device, load_kge_model=None, load_self_model=None, load_rel_model=None, esm_model=None):
        print("load model from disk...")

        if load_kge_model:
            kwargs, state = torch.load(f"{model_path}/../{load_kge_model}", map_location=torch.device(f"{device}"))
            model = KGEModel(**kwargs)
        elif load_self_model:
            kwargs, state = torch.load(f"{model_path}/{load_self_model}", map_location=torch.device(f"{device}"))
            model = SelfModel(kwargs["hidden_dim"], esm_model)
        elif load_rel_model:
            kwargs, state = torch.load(f"{model_path}/{load_rel_model}", map_location=torch.device(f"{device}"))
            model = RelationModel(entity_dim=kwargs["entity_dim"], entity_embeddings=state["entity_embeddings"])
        else:
            raise ValueError

        model.load_state_dict(state, strict=False)
        return model.to(device)









    def train_kge(args):

        # if args.warm_up_steps:
        #     warm_up_steps = args.warm_up_steps
        # else:
        #     warm_up_steps = args.max_relation_steps // 2

        warm_up_steps = args.warm_up_steps


        if args.baseline:
            kg_triples = read_triple(os.path.join(f"{args.data_path}/../", 'drkg.tsv'), entity2id,
                                        relation2id)
            train_triples = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id,
                                        relation2id)
            aug_times = len(kg_triples) // len(train_triples)
            train_triples = train_triples * aug_times + kg_triples
        else:
            train_triples = read_triple(os.path.join(f"{args.data_path}/../", 'drkg.tsv'), entity2id,
                                        relation2id)
        valid_triples = read_triple(os.path.join(args.data_path, 'valid.tsv'), entity2id, relation2id)
        test_triples = read_triple(os.path.join(args.data_path, 'test.tsv'), entity2id, relation2id)
        test_neg_triples = read_triple(os.path.join(args.data_path, 'test_neg.tsv'), entity2id, relation2id)
        valid_neg_triples = read_triple(os.path.join(args.data_path, 'valid_neg.tsv'), entity2id, relation2id)

        valid_labels = torch.cat((torch.ones(len(valid_triples)), torch.zeros(len(valid_neg_triples))))
        test_labels = torch.cat(
            (torch.ones(len(test_triples)), torch.zeros(len(test_neg_triples))))  # do not need it in train

        train_relation_dataloader_head = DataLoader(
            RelationPretrainDataset(train_triples, n_entity, n_relation, args.negative_sample_size, 'head-batch'),
            batch_size=args.relation_batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=RelationPretrainDataset.collate_fn,
        )
        train_relation_dataloader_tail = DataLoader(
            RelationPretrainDataset(train_triples, n_entity, n_relation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.relation_batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=RelationPretrainDataset.collate_fn
        )
        train_relation_iterator = BidirectionalOneShotIterator(train_relation_dataloader_head,
                                                               train_relation_dataloader_tail)
        test_relation_dataloader = DataLoader(
            TestRelationPretrainDataset(
                test_triples + test_neg_triples,
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=TestRelationPretrainDataset.collate_fn,
            shuffle=False
        )
        valid_relation_dataloader = DataLoader(
            TestRelationPretrainDataset(
                valid_triples + valid_neg_triples,
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 20),
            collate_fn=TestRelationPretrainDataset.collate_fn,
            shuffle=False
        )

        kge_model = KGEModel(
            model_name=args.model,
            nentity=n_entity,
            nrelation=n_relation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding
        )
        kge_model = kge_model.to(args.device)

        # Set training configuration
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )



        kge_model, test_acc, test_auc, test_aupr = train_loop(kge_model, optimizer, warm_up_steps, train_relation_iterator, kge_model.train_relation_step,
                   True,
                   valid_relation_dataloader, valid_labels, kge_model.evaluate, "relation", test_relation_dataloader, test_labels)

        return kge_model, test_acc, test_auc, test_aupr


    def gen_psd_label(fun, args, model=None):

        def gen_cartesian_triples(relation2id, ent_id2seq, ent_id2smiles, cartesian_ratio):
            cmp_with_smiles, gene_with_seq = tuple(ent_id2smiles.keys()), tuple(ent_id2seq.keys())
            cmp_gene_cartesian_product = list(itertools.product(cmp_with_smiles, (relation2id["DTI"],), gene_with_seq))
            print(len(cmp_gene_cartesian_product))
            print(cmp_gene_cartesian_product[0:5])
            cmp_gene_cartesian_product = random.sample(cmp_gene_cartesian_product,
                                                       int(len(cmp_gene_cartesian_product) * cartesian_ratio))
            return cmp_gene_cartesian_product

        _, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        del _
        esm_converter = esm_alphabet.get_batch_converter()
        ent_id2seq, ent_id2smiles = prepare_self_esm_data(f"{args.data_path}/../ent_id2seq.csv",
                                                          f"{args.data_path}/../ent_id2smiles.csv", esm_converter)

        cartesian_ratio = args.kge_cartesian_ratio if fun == "psd_label_kge" else args.self_cartesian_ratio
        cartesian_triples = gen_cartesian_triples(relation2id, ent_id2seq, ent_id2smiles, cartesian_ratio)

        test_cartesian_dataloader = DataLoader(
            SelfDataset(
                cartesian_triples, list(), ent_id2smiles, ent_id2seq
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=max(1, args.cpu_num // 100),
            collate_fn=SelfDataset.collate_fn,
        )
        preds = model.evaluate(test_cartesian_dataloader, args, if_metric=False)

        print("End evaluate~~~~~~~~~~~")
        triple_preds = [(cartesian_triples[i], preds[i]) for i in range(len(preds))]
        triple_preds.sort(key=lambda x: x[1], reverse=True)
        if args.save:
            save2file(triple_preds, args.pkl_path, start_time, info=fun)
        return triple_preds



    def train_module(fun, args, triple_preds=None, kge_model=None, self_model=None, rel_model=None, max_ephs=999999):
        def read_psd_triples(true_triple_len, psd_use_ratio, balance_psd, triple_preds, balance_ratio):
            psd_use_num = int(len(triple_preds) * psd_use_ratio)
            psd_triples = list(triple_preds[i][0] for i in range(psd_use_num))
            if balance_psd:
                balance_times = psd_use_num / true_triple_len
                if balance_times < 1:
                    balance_times = 1
                    print("psd triples are fewer than real triples")
                else:
                    balance_times = int(balance_times*balance_ratio)
            else:
                balance_times = 1
            psd_neg_triples = list(random.sample(triple_preds[int(-0.5 * len(triple_preds)):],
                                                 psd_use_num + (balance_times - 1) * true_triple_len))
            psd_neg_triples = list(tri_pred[0] for tri_pred in psd_neg_triples)
            return psd_triples, psd_neg_triples, balance_times



        train_real_triples = read_triple(os.path.join(args.data_path, 'train.tsv'), entity2id,
                                         relation2id)

        if fun in ("train_self", "train_relation"):



            psd_triples, psd_neg_triples, balance_times = read_psd_triples(len(train_real_triples),
                                                                           args.self_psd_use_ratio if fun=="train_relation" else args.rel_psd_use_ratio,
                                                                           args.balance_psd,
                                                                           triple_preds,
                                                                           args.self_balance_ratio if fun == "train_relation" else args.rel_balance_ratio
                                                                           )



            train_triples = (train_real_triples * balance_times) + psd_triples
            train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
                                            relation2id) + psd_neg_triples



        elif fun == "train_projector":
            # train_triples = train_real_triples
            # train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id, relation2id)
            # pass

            psd_triples, psd_neg_triples, balance_times = read_psd_triples(
                                                                           len(train_real_triples), args.gate_psd_use_ratio,
                                                                           args.balance_psd, triple_preds, args.gate_balance_ratio)

            train_triples = (train_real_triples * balance_times) + psd_triples
            train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
                                            relation2id) + psd_neg_triples


        # elif fun in ("train_projector_pure", "train_relation_pure", "train_self_pure"):
            # aug_times = 542 // len(train_real_triples)  # The number 542 has no special meaning. 500, 510, etc are both ok.
            # if aug_times == 0:
            #     train_triples = train_real_triples
            #     train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
            #                                     relation2id)
            # else:
            #     train_triples = train_real_triples * aug_times
            #     train_neg_triples = read_triple(f"{args.data_path}/../full_drugcentral/train_neg.tsv", entity2id, relation2id)
            #     train_neg_triples = train_neg_triples[:len(train_triples)]
        elif fun in ("train_projector_pure", "train_relation_pure", "train_self_pure"):
            if args.aug_pure_train:
                pass
            else:
                train_triples = train_real_triples
                train_neg_triples = read_triple(os.path.join(args.data_path, 'train_neg.tsv'), entity2id,
                                                relation2id)
        else:
            raise ValueError

        valid_triples = read_triple(os.path.join(args.data_path, 'valid.tsv'), entity2id, relation2id)
        test_triples = read_triple(os.path.join(args.data_path, 'test.tsv'), entity2id, relation2id)
        valid_neg_triples = read_triple(os.path.join(args.data_path, 'valid_neg.tsv'), entity2id, relation2id)
        test_neg_triples = read_triple(os.path.join(args.data_path, 'test_neg.tsv'), entity2id, relation2id)

        train_dataloader = DataLoader(SelfDataset(train_triples, train_neg_triples, ent_id2smiles, ent_id2seq),
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=max(1, args.cpu_num // 100),
                                      collate_fn=SelfDataset.collate_fn
                                      )
        valid_dataloader = DataLoader(SelfDataset(valid_triples, valid_neg_triples, ent_id2smiles, ent_id2seq),
                                      batch_size=args.train_batch_size,
                                      shuffle=False,
                                      num_workers=max(1, args.cpu_num // 100),
                                      collate_fn=SelfDataset.collate_fn
                                      )
        test_dataloader = DataLoader(SelfDataset(test_triples, test_neg_triples, ent_id2smiles, ent_id2seq),
                                     batch_size=args.train_batch_size,
                                     shuffle=False,
                                     num_workers=max(1, args.cpu_num // 100),
                                     collate_fn=SelfDataset.collate_fn
                                     )

        print(f"len of train set: {len(train_triples)}")
        print(f"len of train neg set: {len(train_neg_triples)}")

        if fun in ("train_relation", "train_relation_pure"):
            if not rel_model:
                model = RelationModel(args.hidden_dim, kge_model.entity_embedding).to(args.device)
            else:
                model = rel_model

            # if fun == "train_relation_pure":
            #     model.freeze_embeddings()
            # else:
            #     model.unfreeze_embeddings()
            model.freeze_embeddings()

        elif fun in ("train_self", "train_self_pure"):
            model = SelfModel(args.hidden_dim, esm_model).to(args.device)
            # model = SelfFAModel(args.hidden_dim, esm_model).to(args.device)


        else:
            def count_paras(model):
                total_param = 0
                learnable_paras = 0
                print("MODEL DETAILS:\n")
                # print(model)
                for param in model.parameters():
                    # print(param.data.size())
                    if param.requires_grad == True:
                        learnable_paras += np.prod(list(param.data.size()))
                    total_param += np.prod(list(param.data.size()))
                return total_param, learnable_paras

            # print(count_paras(self_model))
            rel_model.freeze_embeddings()
            # print(count_paras(rel_model))
            model = GateModel(rel_model, self_model, args.double_layer).to(args.device)
            # print(count_paras(model))
            # exit(0)  # FOR DEBUG
            if not args.gate_free:
                model.freeze()


        # if "train_projector" in fun:
        #     learning_rate = 1e-2
        # else:
        #     learning_rate = 1e-3
        if "train_projector" in fun and "full" not in args.data_path:
            weight_decay = args.gate_weight_decay
            learning_rate = args.gate_learning_rate
            # if_save_opt = args.save_opt
        else:
            weight_decay = args.module_weight_decay
            learning_rate = args.module_learning_rate
            # if_save_opt = args.save_opt

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        # if fun not in ["train_projector", "train_projector_pure"]:
        #     if_save = args.save_opt
        # else:
        #     if_save = False
        # if_save = args.save_opt  # DEBUG

        if "train_projector" in fun:
            es_pat = 5
        elif "train_self" in fun:
            es_pat = 5
        else:
            es_pat = 2
        es = EarlyStopping(args.model_path, args.save_opt, args.save_thre, args.save_mul, fun, patience=es_pat)



        opt_test_aupr = float("-inf")
        # acc_save, auc_save, aupr_save = float("-inf"), float("-inf"), float("-inf")
        for i in range(max_ephs):
            loss = model.train_epoch(train_dataloader, loss_fn, optimizer, args)
            print(f"Epoch {i} loss: {loss:>7f}")
            acc, auc, aupr = model.evaluate(valid_dataloader, args)
            print(f"valid acc:{acc:.4f}, auc:{auc:.4f}, aupr:{aupr:.4f}")


            if_stop, if_opt = es(aupr, model)
            if (if_opt and "train_projector" in fun) or (if_stop and "train_projector" not in fun):
                test_acc, test_auc, test_aupr = model.evaluate(test_dataloader, args)
                print(f"test acc:{test_acc:.4f}, auc:{test_auc:.4f}, aupr:{test_aupr:.4f}#")
                if test_aupr > opt_test_aupr:
                    opt_test_aupr = test_aupr

            # if if_opt or if_stop:

            if if_stop and max_ephs == 999999:
                break

        if args.save and "train_projector" not in fun:
            save2file(model, args.model_path, start_time, info=fun)
        current_time = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
        print(f"End training at time {current_time}")
        print(f"opt test aupr {opt_test_aupr:.4f}")


        return model, test_acc, test_auc, test_aupr


    def set_random_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seeds(args.seed)


    gc.collect()
    torch.cuda.empty_cache()

    # args.device = auto_select_device(True, args.device_num, list(range(8)), strategy="random")

    args.data_path = f"var_data/{args.dataset}"
    args.model_path = f"var_models/{args.dataset}"
    args.pkl_path = f"var_pkls/{args.dataset}"

    args.adv = True
    args.gate_free = True
    # args.rel_free = True
    args.balance_psd = True
    if args.baseline:
        args.load_kge_model = None
    print("{args.rel_psd_use_ratio} {args.self_psd_use_ratio} {args.kge_cartesian_ratio} {args.self_cartesian_ratio}")
    print(f"{args.rel_psd_use_ratio} {args.self_psd_use_ratio} {args.kge_cartesian_ratio} {args.self_cartesian_ratio}")


    if not args.baseline:
        import esm

    entity2id = dict()
    id2entity = dict()
    with open(os.path.join(f"{args.data_path}/../", 'entities.dict')) as fin:
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    relation2id = dict()
    id2relation = dict()
    with open(os.path.join(f"{args.data_path}/../", 'relations.dict')) as fin:
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            id2relation[int(rid)] = relation

    n_entity = len(entity2id)
    n_relation = len(relation2id)

    if not args.baseline:
        esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        esm_model = esm_model.to(args.device)
        esm_converter = esm_alphabet.get_batch_converter()
        ent_id2seq, ent_id2smiles = prepare_self_esm_data(f"{args.data_path}/../ent_id2seq.csv",
                                                          f"{args.data_path}/../ent_id2smiles.csv", esm_converter)


    if args.load_kge_model:
        print("**************************load kge model******************************")
        kge_model = load_model_to_device(args.model_path, args.device, load_kge_model=args.load_kge_model)
    elif args.load_rel_model or args.load_tuned_rel_model:
        pass
    else:
        print("******************Start training kge************************")
        kge_model, kge_acc, kge_auc, kge_aupr = train_kge(args)

    if args.baseline:
        return kge_acc, kge_auc, kge_aupr

    if args.load_rel_model:
        print("***********************load relation model******************************")
        rel_model = load_model_to_device(args.model_path, args.device, load_rel_model=args.load_rel_model)
    elif args.load_tuned_rel_model:
        pass
    else:
        print("***********************Start train relation projector******************************")
        rel_model, rel_acc, rel_auc, rel_aupr = train_module("train_relation_pure", args, kge_model=kge_model, max_ephs=args.rel_train_ephs)


    if args.kge_psd_filename:
        print("**************************load kge triple preds******************************")
        with open(f"{args.pkl_path}/{args.kge_psd_filename}", "rb") as fin:
            kge_triple_preds = pickle.load(fin)
    elif (args.load_self_model and args.gate_pure_train) or args.pure_train:
        pass
    else:
        print("***********************Start generating psd label kge************************")
        kge_triple_preds = gen_psd_label("psd_label_kge", args, rel_model)


    if not args.load_self_model:
        print("************************Start training self model*****************************")
        if not args.pure_train:
            self_model, _, _, _ = train_module("train_self", args, kge_triple_preds, max_ephs=args.self_train_ephs)
        else:
            self_model, self_acc, self_auc, self_aupr = train_module("train_self_pure", args, max_ephs=args.self_train_ephs)
    else:
        print("**************************load self model******************************")
        self_model = load_model_to_device(args.model_path, args.device, load_self_model=args.load_self_model, esm_model=esm_model)


    if args.pure_train or (args.load_tuned_rel_model and args.gate_pure_train):
        pass
    elif not args.self_psd_filename:
        print("************************Start generating psd label self*****************************")
        self_triple_preds = gen_psd_label("psd_label_self", args, self_model)
    else:
        print("**************************load self triple preds******************************")
        with open(f"{args.pkl_path}/{args.self_psd_filename}", "rb") as fin:
            self_triple_preds = pickle.load(fin)



    if args.load_tuned_rel_model:
        print("***********************load relation model******************************")
        rel_model = load_model_to_device(args.model_path, args.device, load_rel_model=args.load_tuned_rel_model)
    elif args.pure_train:
        pass
    else:
        print("***********************Start tune relation projector******************************")
        rel_model, _, _, _ = train_module("train_relation", args, self_triple_preds, rel_model=rel_model,
                                 max_ephs=args.rel_tune_ephs)


    print("***********************Start train gating model******************************")
    if not args.gate_pure_train and not args.pure_train:
        triple_preds = cut_long_preds(kge_triple_preds, self_triple_preds)
        _, _, _, _ = train_module("train_projector", args, triple_preds=triple_preds, rel_model=rel_model, self_model=self_model, max_ephs=args.gate_train_ephs)
    else:
        _, gate_acc, gate_auc, gate_aupr = train_module("train_projector_pure", args=args, rel_model=rel_model, self_model=self_model, max_ephs=args.gate_train_ephs)



    print("*************************exit*****************************")
    if args.pure_train:
        return (rel_acc, rel_auc, rel_aupr, self_acc, self_auc, self_aupr, gate_acc, gate_auc, gate_aupr)
# TILL HERE

def add_args(parser):
    """
    same performance paras with the python -u kge/kge_e2e.py --do_train --cuda --do_valid --do_test --data_path data_rev/drugcentral --model TransE -n 256 -b 1024 -d 1000 -g 24.0 -a 1.0 -adv --record --valid_steps 50000 -lr 0.0001 --max_steps 150000 --test_batch_size 16 --workspace_path ./record/drugcentral/ --topk 100 -sre --iter_index 0 in the excel
    """
    parser.add_argument('--dataset', default='full_drugbank', type=str)
    parser.add_argument('--model', default='TransE', type=str)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    parser.add_argument('-n', '--negative_sample_size', default=256,
                        type=int)  # only this and new arg field different from run.py before
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    parser.add_argument('-g', '--gamma', default=24, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')  # fill
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('--max_relation_steps', default=999999999999999999, type=int)  # do not limit max steps
    parser.add_argument('--warm_up_steps', default=999999999999999999,
                        type=int)  # TBIMP. let it never enter warm up mode

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=50, type=int)
    parser.add_argument('--relation_valid_steps', default=11061, type=int)  # 10000
    parser.add_argument("--device_num", type=int, default=-1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0)

    parser.add_argument('--train_batch_size', default=16, type=int)  # 16
    parser.add_argument("--relation_batch_size", default=1024, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')  # 512
    parser.add_argument("--load_kge_model", type=str,
                        default="2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--device__6--gate__kge.pth")
    parser.add_argument("--load_self_model", type=str)
    parser.add_argument("--load_rel_model", type=str)  # kge + projector
    parser.add_argument("--load_tuned_rel_model", type=str)
    parser.add_argument("--kge_psd_filename", type=str)
    parser.add_argument("--self_psd_filename", type=str)
    parser.add_argument("--gate_free", action='store_true')
    parser.add_argument("--balance_psd", action="store_true")
    parser.add_argument("--double_layer", action="store_true")
    parser.add_argument("--save_mul", action="store_true")
    parser.add_argument("--save_opt", action="store_true")
    parser.add_argument("--save_thre", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--rel_psd_use_ratio", type=float,
                        default=0.00003)  # 0.0002 sheet1 0.002 few dataset. -240503: 0.0005
    parser.add_argument("--self_psd_use_ratio", type=float,
                        default=0.0003)  # 0.0005 sheet1 train relation. 0.0001 sheet1 train projector. 0.05 few dataset. 0.0005
    parser.add_argument("--gate_psd_use_ratio", type=float, default=0.003)  # -240506: 0.0003
    parser.add_argument("--kge_cartesian_ratio", type=float, default=0.02)  # 0.1 sheet1 0.01 few dataset
    parser.add_argument("--self_cartesian_ratio", type=float, default=0.0002)  # 0.1 sheet1. 0.0002 few dataset
    parser.add_argument('--save', action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--pure_train", action="store_true")
    parser.add_argument("--gate_pure_train", action="store_true")
    parser.add_argument("--rel_train_ephs", type=int, default=999999)  # 3
    parser.add_argument("--self_train_ephs", type=int, default=999999)  # 6 5
    parser.add_argument("--rel_tune_ephs", type=int, default=999999)  # 1
    parser.add_argument("--gate_train_ephs", type=int, default=999999)  # 15  4
    parser.add_argument("--self_balance_ratio", type=float, default=1.0)
    parser.add_argument("--rel_balance_ratio", type=float, default=0.3)  # -240503: 1
    parser.add_argument("--gate_balance_ratio", type=float, default=1.0)  # -240503: 1 -240506: 0.7
    parser.add_argument("--module_weight_decay", type=float, default=0)
    parser.add_argument("--gate_weight_decay", type=float, default=1e-2)
    parser.add_argument('--module_learning_rate', default=1e-3, type=float)
    parser.add_argument('--gate_learning_rate', default=1e-2, type=float)
    parser.add_argument('--aug_pure_train', action="store_true")
    # parser.add_argument("--load_entire_model", type=str)
    return parser


if __name__ == '__main__':


    start_time = str(datetime.datetime.now())[0:-4].replace(' ', '_').replace(":", "_")

    parser = argparse.ArgumentParser()

    parser = add_args(parser)
    args = parser.parse_args()





    main(args, start_time)

