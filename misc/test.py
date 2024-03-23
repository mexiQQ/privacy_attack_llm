import torch
import time

def method_loop(w1, w2, w3):
    d = w1.size(1)
    W_loop = torch.zeros((d, d, d), device='cuda')
    for i in range(d):
        for j in range(d):
            for k in range(d):
                W_loop[i, j, k] = (w1[:, i] * w2[:, j] * w3[:, k]).sum()
    return W_loop

def method_optimized(w1, w2, w3):
    w1_exp = w1.unsqueeze(2).unsqueeze(3)  # [m, d, 1, 1]
    w2_exp = w2.unsqueeze(1).unsqueeze(3)  # [m, 1, d, 1]
    w3_exp = w3.unsqueeze(1).unsqueeze(2)  # [m, 1, 1, d]
    W_optimized = (w1_exp * w2_exp * w3_exp).sum(dim=0)  # [d, d, d]
    return W_optimized

def main():
    torch.manual_seed(0)  # 为了结果的可重复性
    m, d = 10, 100  # 假设的维度，根据你的GPU内存调整
    w1 = torch.randn(m, d, device='cuda')
    w2 = torch.randn(m, d, device='cuda')
    w3 = torch.randn(m, d, device='cuda')

    start_time = time.time()
    W_loop = method_loop(w1, w2, w3)
    loop_time = time.time() - start_time

    start_time = time.time()
    W_optimized = method_optimized(w1, w2, w3)
    optimized_time = time.time() - start_time

    import pdb; pdb.set_trace()
    
    # 验证两种方法的结果是否一致
    print("Are the results from loop and optimized method the same?",
          torch.allclose(W_loop, W_optimized, atol=1e-6))

    print(f"Loop method took {loop_time:.6f} seconds.")
    print(f"Optimized method took {optimized_time:.6f} seconds.")

if __name__ == "__main__":
    main()
