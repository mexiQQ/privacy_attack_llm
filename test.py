import torch
import torch.nn.functional as F
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
import numpy as np
from scipy.spatial import distance
from attack_sheng.tensor_attack import tensor_feature

def compute_grads(model, x_embeds, y_labels, create_graph=False, return_pooler=False, return_first_token_tensor=False, cheat=False, debug=False):
    # outs, ori_pooler_dense_input, pooled_output, pooled_output_before_activation = model(
    #     inputs_embeds=x_embeds, 
    #     labels=y_labels, 
    #     return_first_token_tensor=True,
    #     return_pooled_output=True)

    outs, ori_pooler_dense_input = model(
    inputs_embeds=x_embeds, 
    labels=y_labels, 
    return_first_token_tensor=True,
    return_pooled_output=True)
        
    # 1. a = w1*x, where a is a 30000-dimensional vector.
    # 2. h = f(a) = a^2 + a^3, where h is also a 30000-dimensional vector.
    # 3. z = w2*h, where z is a 2-dimensional vector.

    # 1. ∂L/∂z = softmax(z) - y, where softmax(z) and y are 2-dimensional vectors. Therefore, ∂L/∂z is also a 2-dimensional vector.
    # 2. ∂z/∂h = w2, which is a 30000x2 matrix.
    # 3. ∂z/∂w2 is simply the output of the first layer h due to the linearity of the operation, which is a 30000-dimensional vector.
    # 4. ∂h/∂a = 2a + 3a^2, which is a 30000-dimensional vector as a is.
    # 5. ∂a/∂w1 = x, which is a 768-dimensional vector.

    # y = F.one_hot(y_labels.squeeze(1), num_classes=2)
    # ∂L/∂w2 = ∂L/∂z * ∂z/∂w2.
    # ∂L/∂w2 = (softmax(z) - y) * h^T
    # w2_manual_grad = ((F.softmax(outs.logits, dim=1) - y) * pooled_output.T).T
    # ∂L/∂w1 = ∂L/∂z * ∂z/∂h * ∂h/∂a * ∂a/∂w1.
    # ∂L/∂w1 = (softmax(z) - y) * w2 * (2a + 3a^2) * x.
    # w1_manual_grad = (((F.softmax(outs.logits, dim=1) - y) @ model.classifier.weight.data) * (2 * pooled_output_before_activation + 3 * pooled_output_before_activation ** 2)).T @ ori_pooler_dense_input

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

    m = 30000 - 768
    d = sub_dimension
    B = len(x_embeds)
    
    # if cheat:
    #     return gradients, ori_pooler_dense_input[:, :sub_dimension].detach()
            
    # activarteion
    
    # even => relu => 特殊处理
    # odd => sigmoid => 特殊处理
    # squer + cubic => no 特殊处理
    
    # g = model.classifier.weight.grad.cpu().numpy()[1].reshape(m) #1 x m
    if debug:
        # classification
        # regression
        # 30000 x 2 (:768 x 2) (768:, 2) => 1/30000 => (1/(30000-768))
        # -1 bias -2 weights
        # g = gradients[-2].cpu().numpy()[1][768:].reshape(m)
        g = gradients[-4].cpu().numpy()[768:, :d]
        # => gradient of W => 100
    else:
        g = gradients[-2].cpu().numpy()[1].reshape(m) #1xm
    
    if debug:
        # in 768 x out 30000 (:,:768) (:, 768:) 
        # (100:, 768:) => 0
        # (:100, 768:) => random 
                                                #out   #in
        W = model.bert.pooler.dense.weight.data[768:, :d].cpu().numpy()
    else:
        W = model.bert.pooler.dense.weight.data[:, :d].cpu().numpy() #m, d

    new_recX = tensor_feature(g, W, 100, B, m, d)
    ######################################################################    
    # import pdb; pdb.set_trace()
    highest = [0] * B
    highest_index = [0] * B
    for i in range(B):
        for j in range(B):
            recover = new_recX[:, j:j+1]
            target = ori_pooler_dense_input[i:i+1, :]
            cosine_similarity = check_cosine_similarity_for_1_sample(
                recover,
                target
            )
            if highest[i] < cosine_similarity:
                highest[i] = cosine_similarity
                highest_index[i] = j

        print("*********************************")
        print("*********************************")

    print("average of cosine similarity",sum(highest)/B)
    print("highest_index", highest_index)
    print("highest", highest)

    pooler_target = torch.from_numpy(new_recX.transpose()).cuda()
    pooler_target = pooler_target[torch.tensor(highest_index)]

    return gradients, pooler_target, sum(highest)/B, 0

def check_cosine_similarity_for_1_sample(recover, target):
    ######################################################################
    # 768 => 1
    input = target[:, :100]
    input = input / torch.linalg.norm(input, ord=2, dim=1, keepdim=True)
    input = input.detach().cpu().numpy()

    # 100 => 1
    # new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    new_recXX = recover.transpose() 
    cosin_sim = 1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))
    # print(f"cosin similarity: {cosin_sim}",
    #     f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    
    for i in range(1):
        recover[:, i] = -recover[:, i]
    
    # new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    new_recXX = recover.transpose() 
    cosin_sim = 1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))
    # print(f"cosin similarity: {cosin_sim}",
    #     f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    ######################################################################
    return abs(cosin_sim)
