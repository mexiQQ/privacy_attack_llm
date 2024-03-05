import torch
import torch.nn.functional as F
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
import numpy as np
from scipy.spatial import distance
from b import (
    no_tenfact,
    matlab_eigs,
    matlab_eigs2
)

class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input, target, thresholds=None, margin=0.0):
        cos_sim = self.cosine_similarity(input, target).abs()
        # import pdb; pdb.set_trace()
        
        if thresholds != None: 
            thresholds = torch.tensor(thresholds, device=cos_sim.device)
            # print("origin_cos", cos_sim)
            # print("thresholds", thresholds)

            cos_sim = cos_sim + margin
            cos_sim = torch.where(cos_sim >= thresholds, 1.0, cos_sim)
            # print("updated cos", cos_sim)

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
    grads, pooler_first_token = compute_grads(model, x_embeds, y_labels, create_graph=create_graph, 
        return_first_token_tensor=True)
    if true_pooler is not None:
        input = pooler_first_token[:, :args.rd]
        input = input / torch.linalg.norm(input, ord=2, dim=1, keepdim=True)
        if debug:
            cosine_loss = COSINE_LOSS(input, true_pooler, thresholds, margin=args.coeff_pooler_match_margin)
        else:
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




def compute_pooler(model, x_embeds, y_labels):
    outs, ori_pooler_dense_input = model(
        inputs_embeds=x_embeds, 
        labels=y_labels, 
        return_first_token_tensor=True)
    loss = outs.loss
    loss.backward()

    sub_dimension = 100
    m = 30000
    d = sub_dimension
    B = 1
    Beta = 2
            
    g = model.classifier.weight.grad.cpu().numpy()[1].reshape(m) #1 x m
    W = model.bert.pooler.dense.weight.data[:, :sub_dimension].cpu().numpy() #m, d

    M = np.zeros((d, d))
    aa = np.sum(g)

    for i in range(d): #768
        for j in range(d): #768
            M[i, j] = np.sum(g * W[:, i] * W[:, j])
            if i == j:
                M[i, i] = M[i, i] - aa

    V, D = matlab_eigs(M, Beta)
    WV = W @ V

    T = np.zeros((Beta, Beta, Beta))
    for i in range(Beta):
        for j in range(i, Beta):
            for k in range(j, Beta):
                T[i, j, k] = np.sum(g * WV[:, i] * WV[:, j] * WV[:, k])
                T[i, k, j] = T[i, j, k]
                T[j, i, k] = T[i, j, k]
                T[j, k, i] = T[i, j, k]
                T[k, i, j] = T[i, j, k]
                T[k, j, i] = T[i, j, k]

    for i in range(Beta):
        for j in range(Beta):
            aa = np.sum(g * WV[:, i])
            T[i, j, j] = T[i, j, j] - aa
            T[j, i, j] = T[j, i, j] - aa
            T[j, j, i] = T[j, j, i] - aa

    T = T / m
    rec_X, _, misc = no_tenfact(T, 100, B)
    new_recX = V @ rec_X
    new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()        
    
    ######################################################################
    new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()        
    input = ori_pooler_dense_input[:, :sub_dimension].detach().cpu().numpy()
    print(f"cosin similarity: {1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))}",
        f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    
    for i in range(B):
        new_recX[:, i] = -new_recX[:, i]
    
    new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    input = ori_pooler_dense_input[:, :sub_dimension].detach().cpu().numpy()
    print(f"cosin similarity: {1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))}",
        f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    ######################################################################
    model.zero_grad()
    for param in model.parameters():
        param.grad = None
    return new_recXX

def compute_grads(model, x_embeds, y_labels, create_graph=False, return_pooler=False, return_first_token_tensor=False, cheat=False, debug=False):
    outs, ori_pooler_dense_input = model(
        inputs_embeds=x_embeds, 
        labels=y_labels, 
        return_first_token_tensor=True)
    gradients = torch.autograd.grad(outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True)
    # loss = outs.loss
    # loss.backward()

    if not return_pooler:
        if return_first_token_tensor:
            return gradients, ori_pooler_dense_input
        else:
            return gradients
        
    if debug:
        sub_dimension = 100 
    else:
        sub_dimension = 100 

    m = 30000-768
    d = sub_dimension
    B = 1
    Beta = 1 
    
    if cheat:
        return gradients, ori_pooler_dense_input[:, :sub_dimension].detach()
            
    # activarteion
    # even => even => 特殊处理
    # odd => sigmoid/odd => 特殊处理 
    # squer + cubic => no 特殊处理
    
    # g = model.classifier.weight.grad.cpu().numpy()[1].reshape(m) #1 x m
    if debug:
        # classification
        # regression
        # 30000 x 2 (:768 x 2) (768:, 2) => 1/30000 => (1/(30000-768))
        # -1 bias -2 weights
        g = gradients[-2].cpu().numpy()[1][768:].reshape(m)
        # => gradient of W => 100
    else:
        g = gradients[-2].cpu().numpy()[1].reshape(m) #1xm
    
    if debug:
        # in 768 x out 30000 (:,:768) (:, 768:) 
        # (100:, 768:) => 0
        # (:100, 768:) => random 
                                                #out   #in
        W = model.bert.pooler.dense.weight.data[768:, :100].cpu().numpy() 
    else:
        W = model.bert.pooler.dense.weight.data[:, :sub_dimension].cpu().numpy() #m, d

    M = np.zeros((d, d))
    aa = np.sum(g)

    for i in range(d): #768
        for j in range(d): #768
            M[i, j] = np.sum(g * W[:, i] * W[:, j])
            if i == j:
                M[i, i] = M[i, i] - aa

    V, D = matlab_eigs(M, Beta)
    WV = W @ V

    T = np.zeros((Beta, Beta, Beta))
    for i in range(Beta):
        for j in range(i, Beta):
            for k in range(j, Beta):
                T[i, j, k] = np.sum(g * WV[:, i] * WV[:, j] * WV[:, k])
                T[i, k, j] = T[i, j, k]
                T[j, i, k] = T[i, j, k]
                T[j, k, i] = T[i, j, k]
                T[k, i, j] = T[i, j, k]
                T[k, j, i] = T[i, j, k]

    for i in range(Beta):
        for j in range(Beta):
            aa = np.sum(g * WV[:, i])
            T[i, j, j] = T[i, j, j] - aa
            T[j, i, j] = T[j, i, j] - aa
            T[j, j, i] = T[j, j, i] - aa

    T = T / m
    rec_X, _, misc = no_tenfact(T, 100, B)
    new_recX = V @ rec_X

    ######################################################################
    # 768 => 1
    input = ori_pooler_dense_input[:, :sub_dimension]
    input = input / torch.linalg.norm(input, ord=2, dim=1, keepdim=True)
    input = input.detach().cpu().numpy()
    
    # 100 => 1
    # new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    new_recXX = new_recX.transpose() 
    cosin_sim = 1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))
    print(f"cosin similarity: {cosin_sim}",
        f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    
    if cosin_sim > 0:
        if cosin_sim > 0.9:
            pooler_target = torch.from_numpy(new_recXX).cuda()
            return gradients, pooler_target
        else:
            return gradients, None
    
    for i in range(B):
        new_recX[:, i] = -new_recX[:, i]
    
    # new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    new_recXX = new_recX.transpose() 
    cosin_sim = 1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))
    print(f"cosin similarity: {cosin_sim}",
        f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    ######################################################################
    if cosin_sim > 0.9:
        pooler_target = torch.from_numpy(new_recXX).cuda()
        return gradients, pooler_target
    else:
        return gradients, None

def compute_grads_new(model, x_embeds, y_labels, create_graph=False, return_pooler=False, return_first_token_tensor=False, cheat=False, debug=False):
    from attack_sheng.no_tenfact import no_tenfact

    outs, ori_pooler_dense_input = model(
        inputs_embeds=x_embeds, 
        labels=y_labels, 
        return_first_token_tensor=True)
    gradients = torch.autograd.grad(outs.loss, model.parameters(), create_graph=create_graph, allow_unused=True)

    if not return_pooler:
        if return_first_token_tensor:
            return gradients, ori_pooler_dense_input
        else:
            return gradients
        
    if debug:
        sub_dimension = 100 
    else:
        sub_dimension = 100 
    m = 30000-768
    d = sub_dimension
    B = 1
    Beta = 1
    
    if cheat:
        return gradients, ori_pooler_dense_input[:, :sub_dimension].detach()
            
    # g = model.classifier.weight.grad.cpu().numpy()[1].reshape(m) #1 x m
    if debug:
        g = gradients[-2].cpu().numpy()[1][768:].reshape(m)
    else:
        g = gradients[-2].cpu().numpy()[1].reshape(m) #1xm
    
    if debug:
        W = model.bert.pooler.dense.weight.data[768:, :sub_dimension].cpu().numpy() 
    else:
        W = model.bert.pooler.dense.weight.data[:, :sub_dimension].cpu().numpy() #m, d
        
    M = np.zeros((d, d))
    aa = np.sum(g)

    for i in range(d): #768
        for j in range(d): #768
            M[i, j] = np.sum(g * W[:, i] * W[:, j])
            if i == j:
                M[i, i] = M[i, i] - aa

    V, D = matlab_eigs2(M, Beta)
    WV = W @ V

    T = np.zeros((Beta, Beta, Beta))
    for i in range(Beta):
        for j in range(i, Beta):
            for k in range(j, Beta):
                T[i, j, k] = np.sum(g * WV[:, i] * WV[:, j] * WV[:, k])
                T[i, k, j] = T[i, j, k]
                T[j, i, k] = T[i, j, k]
                T[j, k, i] = T[i, j, k]
                T[k, i, j] = T[i, j, k]
                T[k, j, i] = T[i, j, k]

    for i in range(Beta):
        for j in range(Beta):
            aa = np.sum(g * WV[:, i])
            T[i, j, j] = T[i, j, j] - aa
            T[j, i, j] = T[j, i, j] - aa
            T[j, j, i] = T[j, j, i] - aa

    T = T / m
    rec_X, _, misc = no_tenfact(T, 100, B)
    new_recX = V @ rec_X
    
    ######################################################################
    # 768 => 1
    input = ori_pooler_dense_input[:, :sub_dimension]
    input = input / torch.linalg.norm(input, ord=2, dim=1, keepdim=True)
    input = input.detach().cpu().numpy()
    
    # 100 => 1
    # new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    new_recXX = new_recX.transpose() 
    cosin_sim = 1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))
    print(f"cosin similarity: {cosin_sim}",
        f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    
    if cosin_sim > 0:
        if cosin_sim > 0.9:
            pooler_target = torch.from_numpy(new_recXX).cuda()
            return gradients, pooler_target
        else:
            return gradients, None
    
    for i in range(B):
        new_recX[:, i] = -new_recX[:, i]
    
    # new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    new_recXX = new_recX.transpose() 
    cosin_sim = 1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))
    print(f"cosin similarity: {cosin_sim}",
        f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    ######################################################################
    if cosin_sim > 0.9:
        pooler_target = torch.from_numpy(new_recXX).cuda()
        return gradients, pooler_target
    else:
        return gradients, None