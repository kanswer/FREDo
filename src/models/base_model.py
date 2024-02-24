from src.models.util import process_long_input
from random import sample
import torch
import torch.nn as nn
from transformers import BertConfig, RobertaConfig, DistilBertConfig, XLMRobertaConfig
import numpy as np
import torch.nn.functional as F
from src.models.losses import ATLoss

class BaseEncoder(nn.Module):
    def __init__(self, config, model, exemplar_method, cls_token_id=0, sep_token_id=0, markers=True):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.markers = markers

        self.nota_embeddings = nn.Parameter(torch.zeros((20,1536)))
        self.nota_loc = nn.Parameter(torch.Tensor(20, 768))
        self.weight_word = nn.Parameter(torch.Tensor(768, 1))  
        torch.nn.init.uniform_(self.nota_embeddings, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.nota_loc, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.weight_word, a=-1.0, b=1.0)
        self.first_run = True
        self.cross_domain = False

        self.bias = nn.Parameter(torch.zeros(1))

        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        self.set_exemplars = exemplar_method
        self.rel_glo_linear = nn.Linear(768, 768 * 2)
        self.rel_feature_att = nn.Linear(2304, 2304)
        


    def encode(self, input_ids, attention_mask):
        # Source: https://github.com/wzhouad/ATLOP
        config = self.config
        if type(config) == BertConfig or type(config) == DistilBertConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id]
        elif type(config) == RobertaConfig or type(config) == XLMRobertaConfig:
            start_tokens = [self.cls_token_id]
            end_tokens = [self.sep_token_id, self.sep_token_id]
        sequence_output_glo, sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output_glo, sequence_output, attention

    def forward(self,
                exemplar_input_ids=None,
                exemplar_masks=None,
                exemplar_entity_positions=None,
                exemplar_labels=None,
                query_input_ids=None,
                query_masks=None,
                query_entity_positions=None,
                query_labels=None,
                type_labels=None):

        # -------------- build relation representations from exemplars ------------
        prototypes, lamda = self.set_exemplars(exemplar_input_ids, exemplar_masks, exemplar_entity_positions, exemplar_labels, type_labels, self.weight_word)

        nota_id = [x.index("NOTA") for x in type_labels]

        # -------------- build and match candidate representations from queries ------------

        num_queries = query_input_ids.size(-2)
        batch_size = query_input_ids.size(0)

        # -------------- build labels according to prototypes ------------
        labels = None
        if query_labels is not None:
            labels = []
            for batch_i in range(batch_size):
                labels_in_episode = []
                for i, query in enumerate(query_labels[batch_i]):
                    num_entities_in_query = len(query_entity_positions[batch_i][i])
                    labels_in_query = torch.zeros((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])), device="cuda")
                    labels_in_query[:,:, nota_id[batch_i]] = 1
                    for k in range(num_entities_in_query):
                        labels_in_query[k,k, nota_id[batch_i]] = 0
                    for l_h, l_t, l_r in query:
                        labels_in_query[l_h,l_t, nota_id[batch_i]] = 0
                        labels_in_query[l_h,l_t, type_labels[batch_i].index(l_r)] = 1
                    labels_in_episode.append(labels_in_query)
                labels.append(labels_in_episode)


        sequence_output_glo, sequence_output, attention = self.encode(query_input_ids.view(-1, query_input_ids.size(-1)), query_masks.view(-1, query_masks.size(-1)))
        sequence_output_glo = sequence_output_glo.view(-1, num_queries, sequence_output_glo.size(-1))
        sequence_output = sequence_output.view(-1, num_queries, sequence_output.size(-2), sequence_output.size(-1))

        attention1 = attention.mean(1)
        attention = attention.view(-1, num_queries, attention.size(-3), attention.size(-2), attention.size(-1))
        ins_att_score_q, _ = attention1.max(-1)
        ins_att_score_q = F.softmax(torch.tanh(ins_att_score_q), dim=1).unsqueeze(-1)
        ins_att_score_q = ins_att_score_q.view(-1, num_queries, ins_att_score_q.size(-2), ins_att_score_q.size(-1))
        query_loc = torch.sum(ins_att_score_q * sequence_output, dim=2) #loc

        rel_general = torch.bmm(sequence_output.view(-1, sequence_output.size(-2), sequence_output.size(-1)), self.weight_word.unsqueeze(0).repeat(sequence_output.size(0)*sequence_output.size(1),1,1)).view(sequence_output.size(0),sequence_output.size(1),sequence_output.size(2),1)
        rel_general = F.softmax(torch.tanh(rel_general.squeeze(-1)), dim=-1).unsqueeze(-1)
        support_loc0 = torch.sum(rel_general * sequence_output, dim=2)

        all_matches = []
        loss = 0
        candidates = 0
        for batch_i in range(batch_size):

            entity_embeddings = [[] for _ in query_entity_positions[batch_i]]
            #print(entity_positions)
            matches = [[] for _ in query_entity_positions[batch_i]]

            for i, batch_item in enumerate(query_entity_positions[batch_i]):
                
                num_entities_in_query = len(batch_item)
                predictions_for_query = torch.zeros((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])), device="cuda")
                mask = torch.ones((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])), dtype=torch.bool, device="cuda")
                for k in range(num_entities_in_query):
                    mask[k,k, :] = 0

                for entity in batch_item:
                    mention_embeddings = []
                    for mention in entity:
                        if self.markers:
                            m_e = sequence_output[batch_i,i,mention[0],:]
                        else:
                            m_e = torch.mean(sequence_output[batch_i,i,mention[0]:mention[1],:], 0)
                        mention_embeddings.append(m_e)
                        #print(sequence_output[i,mention[0]:mention[1],:].size(), m_e.size())
                    e_e = torch.mean(torch.stack(mention_embeddings, 0), 0)
                    entity_embeddings[i].append(e_e)
                
                embs = torch.stack(entity_embeddings[i])
                # create matrix with all candidate pairs
                h_entities = torch.unsqueeze(embs, 1)
                h_entities = h_entities.repeat(1, h_entities.size()[0], 1)
                t_entities = h_entities.transpose(0,1)

                entity_indexes = []
                for entity_id in range(embs.size(0)):
                    entity_index = torch.tensor([mention[0] for mention in query_entity_positions[batch_i][i][entity_id]])
                    entity_indexes.append(entity_index)

                h_att = torch.tensor([torch.index_select(attention[batch_i][i], 1, entity_index.to('cuda')).mean(0).mean(0).cpu().detach().numpy() for entity_index in entity_indexes]).to('cuda')
                h_att = h_att.repeat(1, h_att.size()[0], 1)
                t_att = h_att.transpose(0, 1)
                h_att, t_att = h_att.flatten(0, 1), t_att.flatten(0, 1)
                ht_att = (h_att * t_att).unsqueeze(0)
                ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
                entity_loc = torch.sum(ht_att.unsqueeze(-1) * sequence_output[batch_i][i].unsqueeze(0), dim=2).squeeze(0)

                # save shape
                target_shape = h_entities.shape

                # flatten
                h_entities, t_entities = h_entities.flatten(0,1), t_entities.flatten(0,1)

                candidates = torch.cat((torch.cat((h_entities, t_entities), dim=-1), (support_loc0[batch_i][i].repeat(h_entities.size(0), 1) + query_loc[batch_i][i].repeat(h_entities.size(0), 1) + entity_loc)/3), dim = -1) #glo
                # candidates = torch.cat((torch.cat((h_entities, t_entities), dim=-1), (query_loc[batch_i][i].repeat(h_entities.size(0), 1) + entity_loc)/2), dim = -1) #glo

                scores = []

                for class_prototypes, class_lamda in zip(prototypes[batch_i], lamda[batch_i]):
                    class_scores = candidates.unsqueeze(0) * class_prototypes.unsqueeze(1)
                    if class_lamda != None:
                        class_scores = class_lamda * class_scores
                    class_scores = torch.sum(class_scores, dim=-1)
                    class_scores = class_scores.max(dim=0,keepdim=False)[0]
                    scores.append(class_scores)

                scores = torch.stack(scores).swapaxes(0,1)
                
                predictions_binary = ATLoss().get_label(scores.detach(), num_labels=1).view((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])))


                predictions_for_query = scores.view((num_entities_in_query, num_entities_in_query, len(type_labels[batch_i])))
                
                for i_h, h in enumerate(entity_embeddings[i]):
                    for i_t, t in enumerate(entity_embeddings[i]):
                        if i_h == i_t:
                            continue

                        for rt in range(len(prototypes[batch_i])):
                            if rt == nota_id[batch_i]:
                                continue
                            if predictions_binary[i_h, i_t, rt] == 1.0:
                                matches[i].append([i_h,i_t,type_labels[batch_i][rt]])

                # ------- LOSS CALCULATION --------
                if query_labels is not None:
                    loss += ATLoss()(torch.masked_select(predictions_for_query, mask).view(-1, predictions_for_query.size(-1)), torch.masked_select(labels[batch_i][i].float(), mask).view(-1, predictions_for_query.size(-1)))
                    #loss += F.binary_cross_entropy_with_logits(torch.masked_select(predictions_for_query, mask).view(-1, predictions_for_query.size(-1)), torch.masked_select(labels[batch_i][i].float(), mask).view(-1, predictions_for_query.size(-1)))

            all_matches.append(matches)

        if query_labels is not None:
            loss = loss / batch_size
            return all_matches, loss
        return all_matches

