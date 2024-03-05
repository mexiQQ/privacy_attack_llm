import torch
import torch.nn.functional as F
from constants import BERT_CLS_TOKEN, BERT_SEP_TOKEN, BERT_PAD_TOKEN
import numpy as np
from scipy.spatial import distance
from attack_sheng.tensor_attack2 import tensor_feature

def compute_grads(model, x_embeds, y_labels, create_graph=False, return_pooler=False, return_first_token_tensor=False, cheat=False, debug=False, args=None):
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
    
    if cheat:
        return gradients, ori_pooler_dense_input[:, :sub_dimension].detach()
            
    # activation
    # even => relu => 特殊处理
    # odd => sigmoid => 特殊处理
    # squar + cubic => no 特殊处理
    
    if debug:
        # g = model.classifier.weight.grad.cpu().numpy()[1].reshape(m) #1 x m
        # g = gradients[-2].cpu().numpy()[1][768:].reshape(m)
        g = gradients[-4].cpu().numpy()[768:, :d]
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

    V = tensor_feature(g, W, 100, B, m, d)
    V = torch.from_numpy(V).float().cuda()

    return gradients, V 
