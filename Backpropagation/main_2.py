#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dlc_practical_prologue import load_data


# In[2]:


train_input, train_target, test_input, test_target = load_data(one_hot_labels = True, normalize = True)


# In[3]:


psi = 0.9
train_target *= psi
test_target *= psi


# In[4]:


import torch 

def sigma(x):
    return torch.tanh(x)


def dsigma(x):
    return 1-torch.pow(sigma(x), 2)


def loss(y, y_hat): # retuns a number (scalar)
    assert (y.shape==y_hat.shape), "Shape of v and t must have the same dimensions"
    return (y-y_hat).pow(2).sum()         #torch.sum(torch.pow((v-t), 2))


def dloss(y, y_hat):
    return 2*(y-y_hat)  # return the same shape with the inputs         


# In[5]:


# Some parameters 
hidden_size = 50
output_size = 10
eps = 1e-6

W1 = torch.normal(0, eps, size=(hidden_size, train_input.shape[1]))
b1 = torch.normal(0, eps, size=(hidden_size, 1))

W2 = torch.normal(0, eps, size=(output_size, hidden_size))
b2 = torch.normal(0, eps, size=(output_size, 1))


dl_dw1 = torch.empty(W1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(W2.size())
dl_db2 = torch.empty(b2.size())


# In[6]:


print(f"W1 shape = {W1.shape}")
print(f"b1 shape = {b1.shape}")

print(f"W2 shape = {W2.shape}")
print(f"b2 shape = {b2.shape}")

print(f"x input shape = {train_input.shape}")


# In[7]:


# torch.mv()
import ipdb


# In[8]:


def forward_pass(w1, b1, w2, b2, x):
    x0 = x    
    s1 = w1.mm(x0.T) + b1
    x1 = sigma(s1) # [50, 1]
    s2 = w2.mm(x1) + b2
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2 # x2 ouput = [10, 1]


# In[9]:


x0, s1, x1, s2, x2 = forward_pass(W1, b1, W2, b2, train_input[2].view(1, -1)) # x = torch.Size([1, 784])
x2 # torch.Size([10, 1])


# In[10]:


x0, s1, x1, s2, x2 = forward_pass(W1, b1, W2, b2, train_input[2].view(1, -1)) # x = torch.Size([1, 784])
x2 # torch.Size([10, 1])


# In[11]:


def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dl_dw1, dl_db1, dl_dw2, dl_db2):
    x0 = x
    dl_dx2 = dloss(x2.view(t.shape), t)
    dl_ds2 = dsigma(s2.squeeze()) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1.squeeze()) * dl_dx1

#     ipdb.set_trace()
    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2.view(-1, 1))
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1.view(-1, 1))


# In[ ]:


nb_classes = train_target.size(1)       # = output_size
nb_train_samples = train_input.size(0)  
eta = 1e-1 / nb_train_samples           # learning rate 
epocs = 1000                            # number of time to iterate through our data

for k in range(epocs):

    acc_loss = 0
    nb_train_errors = 0

    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()

    for n in range(nb_train_samples):
        x0, s1, x1, s2, x2 = forward_pass(W1, b1, W2, b2, train_input[n].view(1, -1))
        pred = x2.max(0)[1].item()   # tensor.max() return the max value and it's index we want the value of the index
#         ipdb.set_trace()
        if train_target[n, pred] != 0.9: 
            nb_train_errors = nb_train_errors + 1
            
        acc_loss = acc_loss + loss(x2.view(train_target[n].shape), train_target[n])

        backward_pass(W1, b1, W2, b2,
                      train_target[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)

    # Gradient step

    W1 = W1 - eta * dl_dw1
    b1 = b1 - eta * dl_db1
    W2 = W2 - eta * dl_dw2
    b2 = b2 - eta * dl_db2

    # Test error

    nb_test_errors = 0

    for n in range(test_input.size(0)):
        _, _, _, _, x2 = forward_pass(W1, b1, W2, b2, test_input[n].view(1, -1))

        pred = x2.max(0)[1].item()
        if test_target[n, pred] != 0.9: 
            nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))