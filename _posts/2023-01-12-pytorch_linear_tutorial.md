# DeepLearning Pytorch linear function tutorial
Hellow everyone this is "CodingOpera" and my name is "Master.M". Nice to meet you guys. Today I will show you how to use linear function DeepLearning frame work in pytorch. The subject is the correlation between salary and career.  

```python
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from tqdm import tqdm
```
First of all, we have to import some class packages. Before this work, you should download class packages using by "pip"method (e.g. "pip install torch"). If you want to more about this, please refer to this site (https://pypi.org/project/torch/). 


```python
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

x = torch.tensor(x).unsqueeze(dim=1).float()
y = torch.tensor(y).unsqueeze(dim=1).float()
```
For this tutorial, we need some data. The "x"dataset is meaning of "career" and "y"dataset is a "salary". And we have to make "tensor" type of data. The "tensor"is a type of data, like box. The AI engine (pytorch, tensorflow etc) need to this "tensor" type data, for the training. So we change the data type, using by "torch.tensor()".  


```python
plt.title('Annual Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(x, y)
plt.show()
```


    
![png](output_2_0.png)

This graph is showing our data set. When career is 1, the salary is also 1. And we already know the salary when career is 1, 2, 3 ... But what if the career is 2.5? In this case we don't know about actual salary, because we don't have this case of data. Instate of this, we can try to estimate the salary when career is 2.5. This concept is the powerful reason why we using the AI.



```python
model = nn.Linear(in_features=1, out_features=1, bias=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
print(model(x))
```
The AI module has three main things that are model, loss and optimizer. In this post, I'll explain it briefly now. And detail one is posting later.
- model: It is most important thing as the type of AI. This have many type of model like Linear, RNN, LSTM etc. Most of AI engineers update this part.
- loss: The difference between prediction and real data. Most of AI models are trained to reduce the loss.
- optimizer: The way to reduce a loss. It also has many algorithm like Adam, SGD, Adagrad.| 


    tensor([[-0.4567],
            [-0.2023],
            [ 0.0521],
            [ 0.3066],
            [ 0.5610],
            [ 0.8154],
            [ 1.0698],
            [ 1.3242],
            [ 1.5786],
            [ 1.8330]], grad_fn=<AddmmBackward>)
    


```python
for step in tqdm(range(10000)):
    prediction = model(x)
    loss = criterion(input=prediction, target=y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    100%|███████████████████████████████████| 10000/10000 [00:02<00:00, 3886.43it/s]
    


```python
test_x = [1+i*0.01 for i in range(900)]
test_x = torch.tensor(test_x).unsqueeze(dim=1).float()
test_y = model(test_x).detach()
```


```python
plt.title('Annual Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(x, y)
plt.scatter(test_x, test_y, s=0.1, c='r')
plt.legend(('data', 'prediction'))
plt.show()
```


    
![png](output_6_0.png)
    



```python

```
