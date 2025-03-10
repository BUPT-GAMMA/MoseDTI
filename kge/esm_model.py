from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as nnfun
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgllife.model import MLPNodeReadout
from dgl import function as fn

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score




class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.kwargs = {"model_name": model_name, "nentity":nentity, "nrelation":nrelation,
                       "hidden_dim": hidden_dim, "gamma":gamma,
                       "double_entity_embedding": double_entity_embedding,
                       "double_relation_embedding": double_relation_embedding
                       }
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')






    def forward(self, sample, module_name, mode='single'):  # HERE
        """
        train and test dataset can have different neg sample size

        the format of sample can be different in different module and mode
        """
        if mode in ["single", "single-head-batch", "single-tail-batch"]:
            if module_name in ["molecule_train", "protein_train", "fuse_test", "molecule_test"]:
                sample, self_data = sample
            elif module_name == "relation":
                pass
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

            head = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(  # 1024 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            if module_name == "fuse_test":
                head_smiles_batch = self.encode_compounds(self_data["head_smiles_batch"]).unsqueeze(1)
                tail_smiles_batch = self.encode_compounds(self_data["tail_smiles_batch"]).unsqueeze(1)
                head_seq_batch = self.encode_proteins(self_data["head_seq_batch"]).unsqueeze(1)
                tail_seq_batch = self.encode_proteins(self_data["tail_seq_batch"]).unsqueeze(1)

                head[self_data["if_head_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((head[self_data["if_head_smiles_batch"]], head_smiles_batch), dim=2))
                tail[self_data["if_tail_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((tail[self_data["if_tail_smiles_batch"]], tail_smiles_batch), dim=2))
                head[self_data["if_head_seq_batch"]] = self.protein_mlp(
                    torch.cat((head[self_data["if_head_seq_batch"]], head_seq_batch), dim=2))
                tail[self_data["if_tail_seq_batch"]] = self.protein_mlp(
                    torch.cat((tail[self_data["if_tail_seq_batch"]], tail_seq_batch), dim=2))


            elif module_name == "molecule_test":
                head_smiles_batch = self.encode_compounds(self_data["head_smiles_batch"]).unsqueeze(1)
                tail_smiles_batch = self.encode_compounds(self_data["tail_smiles_batch"]).unsqueeze(1)
                head[self_data["if_head_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((head[self_data["if_head_smiles_batch"]], head_smiles_batch), dim=2))
                tail[self_data["if_tail_smiles_batch"]] = self.molecule_mlp(
                    torch.cat((tail[self_data["if_tail_smiles_batch"]], tail_smiles_batch), dim=2))


            elif module_name == "molecule_train":
                if mode == "single-head-batch":
                    self_tail = self.encode_compounds(self_data).unsqueeze(1)
                    tail = self.molecule_mlp(torch.cat((self_tail, tail), 2))
                elif mode == "single-tail-batch":
                    self_head = self.encode_compounds(self_data).unsqueeze(1)
                    head = self.molecule_mlp(torch.cat((self_head, head), 2))
                else:
                    raise ValueError('mode %s not supported' % mode)


            elif module_name == "protein_train":
                if mode == "single-head-batch":
                    self_tail = self.encode_proteins(self_data).unsqueeze(1)
                    tail = self.protein_mlp(torch.cat((self_tail, tail), 2))
                elif mode == "single-tail-batch":
                    self_head = self.encode_proteins(self_data).unsqueeze(1)
                    head = self.protein_mlp(torch.cat((self_head, head), 2))
                else:
                    raise ValueError('mode %s not supported' % mode)




        elif mode == "head-batch":  # positive entities are tails
            if module_name == "relation":
                tail_part, head_part = sample  # 16 * 3, 16 * 97238


                tail = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)

            elif module_name == "molecule_train":
                tail_part, head_part, smiles_batch = sample
                tail = self.encode_compounds(smiles_batch).unsqueeze(1)
                tail_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
                tail = self.molecule_mlp(torch.cat((tail, tail_free), 2))

            elif module_name == "protein_train":
                tail_part, head_part, seq_batch = sample
                tail = self.encode_proteins(seq_batch).unsqueeze(1)
                tail_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
                tail = self.protein_mlp(torch.cat((tail, tail_free), 2))

            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")


            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(  # 16 * 97238 * 1000
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(  # 16 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)


        elif mode == "tail-batch":
            if module_name == "relation":
                head_part, tail_part = sample  # 16 * 3, 16 * 97238

                head = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)

            elif module_name == "molecule_train":
                head_part, tail_part, smiles_batch = sample
                head = self.encode_compounds(smiles_batch).unsqueeze(1)
                head_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
                head = self.molecule_mlp(torch.cat((head, head_free), 2))

            elif module_name == "protein_train":
                head_part, tail_part, seq_batch = sample
                head = self.encode_proteins(seq_batch).unsqueeze(1)
                head_free = torch.index_select(  # 16 * 1 * 1000
                    self.entity_embedding,
                    dim=0,
                    index=head_part[:, 2]
                ).unsqueeze(1)
                head = self.protein_mlp(torch.cat((head, head_free), 2))

            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == "single-head-batch":  # for single mode in self train. entities with self feat are tails.
            sample, self_feat = sample
            head = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(  # 1024 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            if module_name == "molecule_train":  # TODO
                tail = self.encode_compounds(self_feat).unsqueeze(1)
            elif module_name == "protein_train":
                tail = self.encode_proteins(self_feat).unsqueeze(1)
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")


        elif mode == "single-tail-batch":
            sample, self_feat = sample
            relation = torch.index_select(  # 1024 * 1 * 1000
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(  # 1024 * 1 * 1000
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            if module_name == "molecule_train":
                head = self.encode_compounds(self_feat).unsqueeze(1)
            elif module_name == "protein_train":
                head = self.encode_proteins(self_feat).unsqueeze(1)
            else:
                raise ValueError("!!!!!!!!!!!!!!!!The module is not supported!!!!!!!!!!!!!!!")

        else:
            raise ValueError('mode %s not supported' % mode)


        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
            # tail batch
            # 1024 * 1 * 1000
            # 1024 * 1 * 1000
            # 1024 * 256 * 1000
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)  # gamma - ||h+r-t||
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def kge_loss(model, optimizer, positive_score, negative_score, subsampling_weight, args):
        if args.negative_adversarial_sampling:  # true. will amplify good negative sample
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (nnfun.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * nnfun.logsigmoid(-negative_score)).sum(dim=1)  # the neg loss is neged
        else:
            negative_score = nnfun.logsigmoid(-negative_score).mean(dim=1)  # not exactly same with the transe paper

        if args.uni_weight:  # false
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:  # false
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization

        # loss =
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss


    @staticmethod
    def train_relation_step(model, optimizer, train_iterator, args):  # TILL HERE
        """
        A single train step. Apply back-propation and return the loss
        """

        model.train()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator) # it is test that it can iterate as a loop

        positive_sample = positive_sample.to(args.device)
        negative_sample = negative_sample.to(args.device)
        subsampling_weight = subsampling_weight.to(args.device)

        negative_score = model((positive_sample, negative_sample), module_name="relation", mode=mode)

        positive_score = model(positive_sample, module_name="relation", mode="single")  # 1024 * 1

        positive_score = nnfun.logsigmoid(positive_score).squeeze(dim=1)


        return model.kge_loss(model, optimizer, positive_score, negative_score, subsampling_weight, args)


    @staticmethod
    def evaluate(args, model, test_dataloader, labels, if_metric=True):
        """
        Evaluate the model on test or valid datasets
        """
        def correct_count(labels, scores):
            return torch.isclose(scores, labels, atol=0.5, rtol=0).sum().item()



        model.eval()
        preds = list()
        with torch.no_grad():
            for i, sample in enumerate(test_dataloader):
                sample = sample.to(args.device)

                scores = torch.sigmoid(model(sample, module_name="relation", mode="single"))  # sigmoid(gamma - ||h+r-t||)  # TILL HERE
                scores = scores.cpu().squeeze()
                preds += scores.tolist()
                if i % 10000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = correct_count(labels, torch.tensor(preds)) / len(test_dataloader.dataset)
            auc = roc_auc_score(labels.numpy(), preds)
            aupr = average_precision_score(labels.numpy(), preds)
            return preds, acc, auc, aupr
        else:
            return preds


class SelfModel(nn.Module):
    def __init__(self, hidden_dim, esm_model):
        super().__init__()
        self.kwargs = {"hidden_dim": hidden_dim}
        self.entity_dim = hidden_dim
        self.layer_filters_proteins = [1280, 96, 128, self.entity_dim]
        self.cpi_hidden_dim = [78, self.entity_dim, self.entity_dim]

        self.drug_gcn = nn.ModuleList(   #  CHGH
            [GraphConv(in_feats=self.cpi_hidden_dim[i], out_feats=self.cpi_hidden_dim[i + 1]) for i in
             range(len(self.cpi_hidden_dim) - 1)])
        # self.drug_gcn = nn.ModuleList(
        #     [GATConv(in_feats=self.cpi_hidden_dim[i], out_feats=self.cpi_hidden_dim[i + 1], num_heads=1) for i in
        #      range(len(self.cpi_hidden_dim) - 1)])
        self.drug_output_layer = MLPNodeReadout(self.entity_dim, self.entity_dim, self.entity_dim,
                                                activation=nn.ReLU(), mode='max')  # 2-layer mlp and readout

        self.target_cnn = nn.ModuleList(
            [nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i + 1],
                       kernel_size=3, padding=1) for i in range(len(self.layer_filters_proteins) - 1)])

        self.fc = nn.Sequential(nn.Linear(self.entity_dim*2, self.entity_dim), nn.ReLU(), nn.Linear(self.entity_dim, 2))
        self.esm_model = esm_model
        for para in self.esm_model.parameters():
            para.requires_grad = False




    def encode_compounds(self, smiles_batch):
        """
        First embed entities with feature
        """
        compound_graphs, compound_vectors = smiles_batch
        # if len(compound_vectors) > 0:  # up and keep
        #
        #     for l in self.drug_gcn:  # l.weight: float32  The dtype error does not occur every time
        #         compound_vectors = nnfun.relu(l(compound_graphs,
        #                                         compound_vectors))  # compound_graphs has node data x with dtype float64, compound_vector is also float64
        #
        #     compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors))
        # else:
        #     compound_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        for l in self.drug_gcn:  # l.weight: float32  The dtype error does not occur every time
            compound_vectors = nnfun.relu(l(compound_graphs,
                                            compound_vectors))  # compound_graphs has node data x with dtype float64, compound_vector is also float64

        compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors.squeeze()))  # CHGH

        return compound_vectors



    def encode_proteins(self, protein_vectors):
        # if len(protein_seqs) > 0:  # self.entity_dim, down and up
        protein_vectors = self.esm_model(protein_vectors, repr_layers=[33], return_contacts=False)
        protein_vectors = protein_vectors["representations"][33]
        protein_vectors = protein_vectors.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vectors = nnfun.relu(l(protein_vectors))
        protein_vectors = nnfun.adaptive_max_pool1d(protein_vectors, output_size=1)
        protein_vectors = protein_vectors.view(protein_vectors.size(0), -1)
        # else:
        #     protein_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        return protein_vectors



    def forward(self, smiles_embeds, seq_embeds):
        smiles_embeds = self.encode_compounds(smiles_embeds)
        seq_embeds = self.encode_proteins(seq_embeds)
        res = self.fc(torch.cat((smiles_embeds, seq_embeds), 1))
        return res


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
            smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
            seq_feats = seq_feats.to(args.device)
            labels = labels.to(args.device)
            outs = self(smiles_feats, seq_feats)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                out = self(smiles_feats, seq_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        else:
            return all_probs



class SelfFAModel(nn.Module):
    def __init__(self, hidden_dim, esm_model):

        class FALayer(nn.Module):  # no dropout,act_all,act_first in code
            def __init__(self, in_dim, out_dim, dropout, subspace_num, save_model):
                super(FALayer, self).__init__()
                self.dropout_list = nn.ModuleList()
                self.project = nn.Linear(in_dim, out_dim, bias=True)
                self.save_model = save_model
                self.out_dim = out_dim
                self.subspace_num = subspace_num
                self.dim_per_space = out_dim // subspace_num

                self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, self.subspace_num, self.dim_per_space)))
                self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self.subspace_num, self.dim_per_space)))
                self.dropout = nn.Dropout(dropout)

                # for saving alphas
                self.save_alphas = False
                self.alphas = list()

                self.reset_parameters()

            def reset_parameters(self):
                gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_normal_(self.project.weight, gain=gain)
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)

            def degree_mul(self, edges):
                alpha = torch.tanh(edges.data['e'])
                if self.save_alphas:
                    self.alphas.append(alpha)
                alpha_norm = edges.dst['d'].view(-1, 1, 1) * edges.src['d'].view(-1, 1, 1) * alpha
                alpha_norm = self.dropout(alpha_norm)
                edges.data.pop('e')
                return {'alpha_norm': alpha_norm}

            def forward(self, g, h):
                h_projected = self.project(h).view(-1, self.subspace_num, self.dim_per_space)
                el = (h_projected * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (h_projected * self.attn_r).sum(dim=-1).unsqueeze(-1)
                g.ndata.update({'ft': h_projected, 'el': el, 'er': er})
                g.apply_edges(fn.u_add_v('el', 'er', 'e'))
                g.ndata.pop('el')
                g.ndata.pop('er')
                g.apply_edges(self.degree_mul)
                g.update_all(fn.u_mul_e('ft', 'alpha_norm', 'm'),
                             fn.sum('m', 'ft'))
                g.edata.pop('alpha_norm')
                ft = g.ndata.pop('ft')
                ft = ft.view(-1, self.out_dim)
                return ft

        super().__init__()
        self.kwargs = {"hidden_dim": hidden_dim}
        self.entity_dim = hidden_dim
        self.layer_filters_proteins = [1280, 96, 128, self.entity_dim]
        self.cpi_hidden_dim = [78, self.entity_dim, self.entity_dim]

        self.drug_gcn = nn.ModuleList(   #  CHGH
            [FALayer(self.cpi_hidden_dim[i], self.cpi_hidden_dim[i + 1], 0, 1, False) for i in
             range(len(self.cpi_hidden_dim) - 1)])
        # self.drug_gcn = nn.ModuleList(
        #     [GATConv(in_feats=self.cpi_hidden_dim[i], out_feats=self.cpi_hidden_dim[i + 1], num_heads=1) for i in
        #      range(len(self.cpi_hidden_dim) - 1)])
        self.drug_output_layer = MLPNodeReadout(self.entity_dim, self.entity_dim, self.entity_dim,
                                                activation=nn.ReLU(), mode='max')  # 2-layer mlp and readout

        self.target_cnn = nn.ModuleList(
            [nn.Conv1d(in_channels=self.layer_filters_proteins[i], out_channels=self.layer_filters_proteins[i + 1],
                       kernel_size=3, padding=1) for i in range(len(self.layer_filters_proteins) - 1)])

        self.fc = nn.Sequential(nn.Linear(self.entity_dim*2, self.entity_dim), nn.ReLU(), nn.Linear(self.entity_dim, 2))
        self.esm_model = esm_model
        for para in self.esm_model.parameters():
            para.requires_grad = False




    def encode_compounds(self, smiles_batch):
        """
        First embed entities with feature
        """
        compound_graphs, compound_vectors = smiles_batch
        if len(compound_vectors) > 0:  # up and keep

            for l in self.drug_gcn:  # l.weight: float32  The dtype error does not occur every time
                compound_vectors = nnfun.relu(l(compound_graphs,
                                                compound_vectors))  # compound_graphs has node data x with dtype float64, compound_vector is also float64

            compound_vectors = nnfun.relu(self.drug_output_layer(compound_graphs, compound_vectors.squeeze()))
        else:
            compound_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        return compound_vectors



    def encode_proteins(self, protein_vectors):
        # if len(protein_seqs) > 0:  # self.entity_dim, down and up
        protein_vectors = self.esm_model(protein_vectors, repr_layers=[33], return_contacts=False)
        protein_vectors = protein_vectors["representations"][33]
        protein_vectors = protein_vectors.permute(0, 2, 1)
        for l in self.target_cnn:
            protein_vectors = nnfun.relu(l(protein_vectors))
        protein_vectors = nnfun.adaptive_max_pool1d(protein_vectors, output_size=1)
        protein_vectors = protein_vectors.view(protein_vectors.size(0), -1)
        # else:
        #     protein_vectors = torch.zeros(0, self.entity_dim).to(self.device)

        return protein_vectors



    def forward(self, smiles_embeds, seq_embeds):
        smiles_embeds = self.encode_compounds(smiles_embeds)
        seq_embeds = self.encode_proteins(seq_embeds)
        res = self.fc(torch.cat((smiles_embeds, seq_embeds), 1))
        return res


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
            smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
            seq_feats = seq_feats.to(args.device)
            labels = labels.to(args.device)
            outs = self(smiles_feats, seq_feats)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, _, _) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                out = self(smiles_feats, seq_feats)
                all_probs += nnfun.softmax(out, dim=1)[:, 1].cpu().tolist()
                all_classes += out.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        else:
            return all_probs




class RelationModel(nn.Module):
    def __init__(self, entity_dim, entity_embeddings):  # suspect: embedding re-saved cannot be added to statedict
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(entity_dim * 2, entity_dim), nn.ReLU(),
                                nn.Linear(entity_dim, 2))
        self.entity_embeddings = nn.Parameter(entity_embeddings)
        self.kwargs = {
            "entity_dim": entity_dim,
        }

    def forward(self, heads, tails):
        heads = self.entity_embeddings[heads]
        tails = self.entity_embeddings[tails]
        res = self.fc(torch.cat((heads, tails), 1))
        return res

    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (_, _, labels, heads, tails) in enumerate(data_loader):
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            outs = self(heads, tails)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss


    def evaluate(self, data_loader, args, if_metric=True):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (_, _, labels, heads, tails) in enumerate(data_loader):
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                outs = self(heads, tails)
                all_probs += nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist()
                all_classes += outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        else:
            return all_probs


    def freeze_embeddings(self):
        self.entity_embeddings.requires_grad = False

    def unfreeze_embeddings(self):
        self.entity_embeddings.requires_grad = True


class GateModel(nn.Module):
    def __init__(self, relation_model, self_model, if_double_layer):
        super().__init__()
        self.relation_model = relation_model
        self.self_model = self_model
        self.kwargs = {"if_double_layer": if_double_layer}
        hid_dim = self.self_model.entity_dim
        if if_double_layer:
            self.fc = nn.Sequential(nn.Linear(hid_dim*4, hid_dim), nn.ReLU(),
                                    nn.Linear(hid_dim, 2))  # from dive into dl p.201, only the last layer does not need relu
        else:
            self.fc = nn.Linear(hid_dim*4, 2)
        self.relation_model.freeze_embeddings()


    def freeze(self):
        for para in self.relation_model.parameters():
            para.requires_grad = False
        for para in self.self_model.parameters():
            para.requires_grad = False





    def forward(self, heads, tails, smiles_embeds, seq_embeds):
        smiles_embeds = self.self_model.encode_compounds(smiles_embeds)
        seq_embeds = self.self_model.encode_proteins(seq_embeds)
        heads = self.relation_model.entity_embeddings[heads]
        tails = self.relation_model.entity_embeddings[tails]
        weight = self.fc(torch.cat((heads, tails, smiles_embeds, seq_embeds), 1)).softmax(1)
        res_self = self.self_model.fc(torch.cat((smiles_embeds, seq_embeds), 1))
        # print(res_self)  # DEBUG
        res_relation = self.relation_model.fc(torch.cat((heads, tails), 1))  # TILL HERE
        # print(res_relation)  # DEBUG
        res = weight[:,0].unsqueeze(1) * res_self + weight[:,1].unsqueeze(1) * res_relation

        return res


    def train_epoch(self, data_loader, loss_fn, optimizer, args):
        self.train()
        epoch_loss = 0
        for batch_idx, (smiles_feats, seq_feats, labels, heads, tails) in enumerate(data_loader):
            smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
            seq_feats = seq_feats.to(args.device)
            heads = heads.to(args.device)
            tails = tails.to(args.device)
            labels = labels.to(args.device)
            outs = self(heads, tails, smiles_feats, seq_feats)
            loss = loss_fn(outs, labels)
            epoch_loss += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return epoch_loss

    def evaluate(self, data_loader, args, if_metric=True, if_probs=False):
        self.eval()
        test_loss, all_probs, all_classes, all_labels = 0, list(), list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, heads, tails) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                outs = self(heads, tails, smiles_feats, seq_feats)
                if if_probs:
                    print(f"batch {i}:", end=" ")
                    print(nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist())
                all_probs += nnfun.softmax(outs, dim=1)[:, 1].cpu().tolist()
                all_classes += outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            acc = accuracy_score(all_labels, all_classes)
            auc = roc_auc_score(all_labels, all_probs)
            aupr = average_precision_score(all_labels, all_probs)
            return acc, auc, aupr
        else:
            return all_probs


    def evaluate_experts(self, data_loader, args, if_metric=True):
        self.eval()
        self_all_probs, self_all_classes, all_labels = list(), list(), list()
        rel_all_probs, rel_all_classes = list(), list()
        with torch.no_grad():
            for i, (smiles_feats, seq_feats, labels, heads, tails) in enumerate(data_loader):
                smiles_feats = (smiles_feats[0].to(args.device), smiles_feats[1].to(args.device))
                seq_feats = seq_feats.to(args.device)
                heads = heads.to(args.device)
                tails = tails.to(args.device)
                self_outs = self.self_model(smiles_feats, seq_feats)
                rel_outs = self.relation_model(heads, tails)
                self_all_probs += nnfun.softmax(self_outs, dim=1)[:, 1].cpu().tolist()
                rel_all_probs += nnfun.softmax(rel_outs, dim=1)[:, 1].cpu().tolist()
                self_all_classes += self_outs.argmax(1).cpu().tolist()
                rel_all_classes += rel_outs.argmax(1).cpu().tolist()
                all_labels += labels.tolist()
                if i % 30000 == 0 and not if_metric:
                    print(f"batch {i} finished") # FOR TEST

        if if_metric:
            self_acc = accuracy_score(all_labels, self_all_classes)
            rel_acc = accuracy_score(all_labels, rel_all_classes)
            self_auc = roc_auc_score(all_labels, self_all_probs)
            rel_auc = roc_auc_score(all_labels, rel_all_probs)
            self_aupr = average_precision_score(all_labels, self_all_probs)
            rel_aupr = average_precision_score(all_labels, rel_all_probs)
            return self_acc, self_auc, self_aupr, rel_acc, rel_auc, rel_aupr
        else:
            return self_all_probs, rel_all_probs

