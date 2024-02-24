import json
from torch.nn import functional as F
import torch
import torch.nn as nn
from src.models.base_model import BaseEncoder
import math
class Encoder(BaseEncoder):
    def __init__(self, config, model, cls_token_id=0, sep_token_id=0, markers=True):
        super().__init__(config=config, model=model, exemplar_method=self.proto_mnav, cls_token_id=cls_token_id, sep_token_id=sep_token_id, markers=markers)

    def proto_mnav(self,
                input_ids=None,
                attention_mask=None,
                entity_positions=None,
                labels=None, 
                type_labels=None,
                weight_word=None):

        num_exemplars = input_ids.size(-2)
        batch_size = input_ids.size(0)
        max_len = input_ids.size(-1)

        sequence_output_glo, sequence_output, attention = self.encode(input_ids.view(-1, input_ids.size(-1)), attention_mask.view(-1, attention_mask.size(-1)))
        sequence_output_glo = sequence_output_glo.view(-1, num_exemplars, sequence_output_glo.size(-1))
        sequence_output = sequence_output.view(-1, num_exemplars, sequence_output.size(-2), sequence_output.size(-1))
        attention = attention.view(-1, num_exemplars, attention.size(-3), attention.size(-2), attention.size(-1))

        batch_exemplars = []
        batch_label_ids = []
        batch_label_types = []
        batch_entity_loc = []
        for batch_i in range(batch_size):
            episode_label_ids = []
            episode_label_types = []
            entity_embeddings = [[] for _ in entity_positions[batch_i]]
            
            relation_embeddings = []
            entity_locs = []
            label_ids, label_types = [], []
            for batch_item in labels[batch_i]:
                li_in_batch = []
                lt_in_batch = []
                for l_h, l_t, l_r in batch_item:
                    li_in_batch.append((l_h, l_t))
                    lt_in_batch.append(l_r)
                label_ids.append(li_in_batch)
                label_types.append(lt_in_batch)
            
            rts = []
            
            for i, batch_item in enumerate(entity_positions[batch_i]):
                for entity in batch_item:
                    mention_embeddings = []
                    for mention in entity:
                        if self.markers:
                            m_e = sequence_output[batch_i,i,mention[0],:]
                        else:
                            m_e = torch.mean(sequence_output[batch_i,i,mention[0]:mention[1],:], 0)
                        mention_embeddings.append(m_e)

                    e_e = torch.mean(torch.stack(mention_embeddings, 0), 0)

                    entity_embeddings[i].append(e_e)
            
                for i_h, h in enumerate(entity_embeddings[i]):
                    for i_t, t in enumerate(entity_embeddings[i]):
                        if i_h == i_t:
                            continue

                        if (i_h, i_t) in label_ids[i]:
                            episode_label_ids.append(len(relation_embeddings))
                            types_for_label = []
                            for li, lt in zip(label_ids[i], label_types[i]):
                                if li == (i_h, i_t):
                                    types_for_label.append(lt)
                                    rts.append(lt)
                            episode_label_types.append(types_for_label)
                        else:
                            episode_label_ids.append(len(relation_embeddings))
                            episode_label_types.append(["NOTA"])
                        relation_embeddings.append(torch.cat([h, t])) #global
                        #entity attention2: support_loc2
                        # support_support = torch.bmm(sequence_output[batch_i][i].unsqueeze(0), torch.transpose(sequence_output[batch_i][i], -1, -2).unsqueeze(0))
                        #index head and tail entity
                        h_index = torch.tensor([mention[0] for mention in entity_positions[batch_i][i][i_h]])
                        t_index = torch.tensor([mention[0] for mention in entity_positions[batch_i][i][i_t]])

                        h_att = torch.index_select(attention[batch_i][i], 1, h_index.to('cuda')).mean(0).mean(0)
                        t_att = torch.index_select(attention[batch_i][i], 1, t_index.to('cuda')).mean(0).mean(0)
                        ht_att = (h_att * t_att).unsqueeze(0)
                        ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5) 
                        entity_loc = torch.sum(ht_att.unsqueeze(-1) * sequence_output[batch_i][i].unsqueeze(0), dim=1).squeeze(0)
                        entity_locs.append(entity_loc)
            
            batch_exemplars.append(torch.stack(relation_embeddings, 0))
            batch_label_ids.append(episode_label_ids)
            batch_label_types.append(episode_label_types)
            batch_entity_loc.append(torch.stack(entity_locs, 0))
        
        # create prototype embeddings
        relation_des = json.load(open("cache/cached-relation-des.json"))
        batch_prototypes = []
        #work-2-2 增加lamda,
        batch_lamda = [] #每种关系的特征注意力值,维度同实例大小，1536
        n_nota_samples = 20
        for entity_loc, batch_id, exemplars, label_ids, label_types, type_index in zip(batch_entity_loc, range(batch_size), batch_exemplars, batch_label_ids, batch_label_types, type_labels):
            episodes_prototypes = [None for _ in type_index]
            episodes_lamda = [None for _ in type_index]
            # print(label_types)
            for relation_type in type_index:
                embeddings = []
                loc_embeddings = []
                for i, t in zip(label_ids, label_types):
                    if relation_type in t:
                        embeddings.append(exemplars[i])
                        loc_embeddings.append(entity_loc[i])
                embeddings = torch.stack(embeddings, 0)
                loc_embeddings = torch.stack(loc_embeddings, 0)
                if relation_type != "NOTA":
                    # glo relation text: rel_text_glo
                    rel_text = relation_des[relation_type]
                    rel_text_token = [rel_text + [0] * (max_len - len(rel_text))]
                    rel_text_mask = [[1.0] * len(rel_text) + [0.0] * (max_len - len(rel_text))]
                    rel_text_token = torch.tensor(rel_text_token, dtype=torch.long).to('cuda')
                    rel_text_mask = torch.tensor(rel_text_mask, dtype=torch.float).to('cuda')
                    rel_glo, rel_local, attention = self.encode(rel_text_token.view(-1, rel_text_token.size(-1)), rel_text_mask.view(-1, rel_text_mask.size(-1)))
                    embeddings = torch.mean(embeddings, 0, keepdim=True) + self.rel_glo_linear(rel_glo)
                    # ---

                    #local prototypes | local:rel_local, sequence_local
                    #general attention: support_loc0
                    rel_general = torch.bmm(sequence_output[batch_id], weight_word.unsqueeze(0))
                    rel_general = F.softmax(torch.tanh(rel_general.squeeze(-1)), dim=-1).unsqueeze(-1)
                    support_loc0 = torch.sum(rel_general * sequence_output[batch_id], dim=1)
                    # relation attention1: support_loc1
                    rel_support = torch.bmm(sequence_output[batch_id], torch.transpose(rel_local, -1, -2).repeat(sequence_output.size(1),1,1))
                    ins_att_score_s, _ = rel_support.max(-1)
                    ins_att_score_s = F.softmax(torch.tanh(ins_att_score_s), dim=1).unsqueeze(-1)
                    support_loc1 = torch.sum(ins_att_score_s * sequence_output[batch_id], dim=1)
                    ins_att_score_r, _ = rel_support.max(1)
                    ins_att_score_r = F.softmax(torch.tanh(ins_att_score_r), dim=1).unsqueeze(-1)
                    rel_text_loc = torch.sum(ins_att_score_r * rel_local, dim=1)
                    rel_text_loc = torch.mean(rel_text_loc, 0).view(-1, rel_text_loc.size(-1))
                    #entity attention2: loc_embeddings
                    support_loc2 = torch.mean(loc_embeddings, 0, keepdim=True)
                    proto_loc = torch.mean((support_loc0 + support_loc1 + support_loc2)/3, 0, keepdim=True) + rel_text_loc
                    # proto_loc = torch.mean((support_loc1 + support_loc2)/2, 0, keepdim=True) + rel_text_loc
                    # ---
                    embeddings = torch.cat((embeddings, proto_loc), dim=-1)
                    embeddings_embeddings = torch.sum(torch.matmul(embeddings.t(), embeddings), 1) #[k]
                    embeddings_embeddings = F.relu(embeddings_embeddings)
                    lamda = F.relu(self.rel_feature_att(embeddings_embeddings))
                    # embeddings_embeddings = torch.div(embeddings_embeddings, math.sqrt(768))
                    # lamda = F.relu(embeddings_embeddings)
                    # lamda = F.softmax(torch.tanh(embeddings_embeddings), dim=0)
                    episodes_lamda[type_index.index(relation_type)] = lamda
                    embeddings = torch.mean(embeddings, 0, keepdim=True)
                else:
                    if self.first_run and self.training:
                        self.nota_embeddings.data = torch.mean(embeddings, 0, keepdim=True)
                        indexes = torch.randperm(embeddings.shape[0])
                        self.nota_embeddings.data = embeddings[indexes[:n_nota_samples], :]
                        self.first_run = False
                    embeddings = torch.cat((self.nota_embeddings, self.nota_loc), dim=-1)
                episodes_prototypes[type_index.index(relation_type)] = embeddings

            # episodes_prototypes = torch.stack(episodes_prototypes, 0)
            batch_prototypes.append(episodes_prototypes)
            batch_lamda.append(episodes_lamda)
        
        return batch_prototypes, batch_lamda
