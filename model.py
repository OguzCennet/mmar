import torch
import torch.nn as nn
import random
from utils import *


class Width(nn.Module):
    bins = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,35]

    def __init__(self, distance_dim):
        super().__init__()
        self.dim = distance_dim
        self.embeds = nn.Sequential(nn.Embedding(len(self.bins)+1, distance_dim),nn.Dropout(0.20))

    def forward(self, *args):
        x = to_cuda(self.stoi(*args))
        x = self.embeds(x)
        return x

    def stoi(self, lengths):
        return torch.tensor([sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False)


class Distance(nn.Module):
    bins = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,35]

    def __init__(self, distance_dim):
        super().__init__()
        self.dim = distance_dim
        self.embeds = nn.Sequential(nn.Embedding(len(self.bins)+1, distance_dim), nn.Dropout(0.20))

    def forward(self, *args):
        x = to_cuda(self.stoi(*args))
        x = self.embeds(x)
        return x

    def stoi(self, l):
        return torch.tensor([sum([True for i in self.bins if n >= i]) for n in l], requires_grad=False)       



class ResolutionModel(nn.Module):
    """ Mention scoring module"""
    def __init__(self):
        super().__init__()
        self.width = Width(50)
        self.distance = Distance(300)

        self.mmoldal_pair_score = nn.Sequential(nn.Linear(6594, 300),nn.ReLU(),nn.Dropout(0.20),
                                                nn.Linear(300, 1))


    def forward(self, sample):

        recipe_span_vectors = sample.language_vectors['recipe']
        recipe_step_vectors = sample.language_vectors
        recipe_clip_vectors = sample.visual_vectors

        span_vectors = []
        frame_vectors = []
        
        for span in sample.spans:
            if span.index == 0: 
                span_vectors.append(torch.zeros(1536))
                frame_vectors.append(torch.zeros(768))
            else:
                span_xlm_roberta_start = torch.tensor(recipe_span_vectors[span.i1]).float()
                span_xlm_roberta_end = torch.tensor(recipe_span_vectors[span.i2]).float()
                span_vectors.append(torch.cat((span_xlm_roberta_start, span_xlm_roberta_end)))
                frame_vectors.append(torch.tensor(recipe_clip_vectors[span.step_id]))
            
        span_tensor_vectors= to_cuda(torch.stack(span_vectors))
        span_frame_vectors = to_cuda(torch.stack(frame_vectors))
        #print(span_frame_vectors.shape, span_tensor_vectors.shape)


        widthes = self.width([len(s.string.split()) for s in sample.spans])
        span_features = torch.cat((span_tensor_vectors, widthes), dim=1)
        
        anaphor_ids, antecedent_ids, step_distances, resolution_labels = zip(*sample.recipe_pairs)
        resolution_labels = to_cuda(torch.tensor(resolution_labels).float())
        
        distances = self.distance(step_distances)
        anaphor_ids = to_cuda(torch.tensor(anaphor_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        i_span_text = torch.index_select(span_features, 0, anaphor_ids)
        j_span_text = torch.index_select(span_features, 0, antecedent_ids)
       
        ij_span = torch.cat((i_span_text, j_span_text), dim=1)        
        ij_span_cos = i_span_text * j_span_text
        
        i_span_image = torch.index_select(span_frame_vectors, 0, anaphor_ids)
        j_span_image = torch.index_select(span_frame_vectors, 0, antecedent_ids)
        ij_image_cat = torch.cat((i_span_image, j_span_image), dim=1) 

        pair_representation = torch.cat((ij_span, ij_span_cos, distances, ij_image_cat), dim=1)
        reference_scores = self.mmoldal_pair_score(pair_representation)
            
        assert reference_scores.size(0) == antecedent_ids.size(0), "There is a problem!!!"
        split_scores = list(torch.split(reference_scores, sample.antecedent_lenghts, dim=0))

        resolution_sigmoids = [torch.sigmoid(tensr) for tensr in split_scores]
        resolution_probabilities = torch.cat((resolution_sigmoids), dim=0)

        return resolution_sigmoids, resolution_probabilities, resolution_labels
        