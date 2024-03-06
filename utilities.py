import torch
import torch.nn.functional as F
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
import numpy as np
from scipy.spatial import distance

class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input, target, thresholds=None, margin=0.0):
        cos_sim = self.cosine_similarity(input, target).abs()
        
        if thresholds != None: 
            thresholds = torch.tensor(thresholds, device=cos_sim.device)

            cos_sim = cos_sim + margin
            cos_sim = torch.where(cos_sim >= thresholds, 1.0, cos_sim)

        loss = 1 - (cos_sim ** 2).mean()  # You can also use: (1 - cos_sim).mean()
        return loss
    
COSINE_LOSS = CosineSimilarityLoss()
 
def grad_dist(grads1, grads2, args):
    ret = 0.0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            if args.loss == 'cos':
                ret += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2))
            elif args.loss == 'dlg':
                ret += (g1 - g2).square().sum()
            elif args.loss == 'tag':
                ret += (g1 - g2).square().sum() + args.tag_factor * torch.abs(g1 - g2).sum()
            else:
                assert False
            n_g += 1
    if args.loss == 'cos':
        ret /= n_g
    return ret


def get_closest_tokens(inputs_embeds, unused_tokens, embeddings_weight, metric='cos'):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == 'l2':
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == 'cos':
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    d[:, :, unused_tokens] = 1e9
    return d, d.min(dim=2)[1]

# global_door = [False]
def get_reconstruction_loss(model, x_embeds, y_labels, true_grads, args, create_graph=False, true_pooler=None, debug=False, thresholds=None):
    grads, pooler_first_token = compute_grads(model, x_embeds, y_labels, args, create_graph=create_graph, 
        return_first_token_tensor=True)
    if true_pooler is not None:
        input = pooler_first_token[:, :args.rd]
        input = input / torch.linalg.norm(input, ord=2, dim=1, keepdim=True)
        cosine_loss = COSINE_LOSS(input, true_pooler, thresholds, margin=args.coeff_pooler_match_margin)
        
        # cosine_loss = torch.tensor(0.0)
        # cosine_loss = torch.maximum(cosine_loss - 0.1, torch.tensor(0.0))

        # global global_door
        # lower_bound = 0.1
        # upper_bound = 0.2
        # if cosine_loss < lower_bound:
        #     global_door[0] = True

        # if cosine_loss > upper_bound:
        #     global_door[0] = False
        
        # if global_door[0]:
        #    csine_loss = torch.tensor(0.0)

        gradient_loss = grad_dist(true_grads, grads, args)
        return gradient_loss, cosine_loss
    else:
        return grad_dist(true_grads, grads, args), 0 


def get_reconstruction_loss_new(model, x_embeds, y_labels, true_grads, args, create_graph=False, projection=None):
    grads, pooler_first_token = compute_grads(model, x_embeds, y_labels, create_graph=create_graph, 
        return_first_token_tensor=True)
    if projection is not None:
        input = pooler_first_token[:, :args.rd]
        input = input / torch.linalg.norm(input, ord=2, dim=1, keepdim=True)
        
        X = input.T
        P = torch.matmul(torch.matmul(X, torch.linalg.inv(torch.matmul(X.T,X))), X.T) 
        feat_reg = ((projection - torch.matmul(P, projection))**2).mean()

        gradient_loss = grad_dist(true_grads, grads, args)
        return gradient_loss, feat_reg
    else:
        return grad_dist(true_grads, grads, args), 0 


def get_perplexity(gpt2, x_embeds, bert_embeddings_weight, gpt2_embeddings_weight, c=0.1):
    gpt2_embeddings_weight = gpt2_embeddings_weight.repeat(x_embeds.shape[0], 1, 1)

    # Get alphas on BERT embeddings --> transfer to GPT-2
    alpha, _ = get_closest_tokens(x_embeds, bert_embeddings_weight)
    # alpha = torch.cdist(x_embeds[:, :-1, :], bert_embeddings_weight, p=2)
    alpha = F.softmax(-alpha/c, dim=2)
    gpt2_embeds = alpha.bmm(gpt2_embeddings_weight)

    # Pass through GPT-2 and get average perplexity
    out_gpt2 = gpt2(inputs_embeds=gpt2_embeds)
    log_probs = out_gpt2.logits.log_softmax(dim=2)
    fuzzy_perplexity = -(log_probs[:, :-1, :] * alpha[:, 1:, :]).sum(dim=2).mean(dim=1).sum()
    return fuzzy_perplexity


def fix_special_tokens(x_embeds, bert_embeddings_weight, pads):
    x_embeds.data[:, 0] = bert_embeddings_weight[BERT_CLS_TOKEN]
    if pads is not None:
        for sen_id in range(x_embeds.shape[0]):
            x_embeds.data[sen_id, pads[sen_id]:] = bert_embeddings_weight[BERT_PAD_TOKEN]
            x_embeds.data[sen_id, pads[sen_id]-1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    elif x_embeds.shape[0] == 1:
        x_embeds.data[:, -1] = bert_embeddings_weight[BERT_SEP_TOKEN]
    return x_embeds


def remove_padding(tokenizer, ids):
    for i in range(ids.shape[0] - 1, -1, -1):
        if ids[i] == BERT_SEP_TOKEN:
            ids = ids[:i+1]
            break
    return tokenizer.decode(ids)

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

def compute_grads(model, x_embeds, y_labels, args, create_graph=False, return_first_token_tensor=False):
    outs, ori_pooler_dense_input = model(
        inputs_embeds=x_embeds, 
        labels=y_labels, 
        return_first_token_tensor=True)
    gradients = torch.autograd.grad(outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True)

    if return_first_token_tensor:
        return gradients, ori_pooler_dense_input
    else:
        return gradients