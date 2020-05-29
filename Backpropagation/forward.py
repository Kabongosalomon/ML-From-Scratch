from dlc_practical_prologue import sigma

def forward_pass(w1, b1, w2, b2, x):
    x0 = x    
    s1 = w1.mm(x0.T) + b1
    x1 = sigma(s1) # [50, 1]
    s2 = w2.mm(x1) + b2
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2 # x2 ouput = [10, 1]