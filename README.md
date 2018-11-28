
# t2t_trainer in eager model



## Run



```bash
git clone https://github.com/xu-song/tensor2tensor-eager.git
cd tensor2tensor-eager
python train_wmt_eager.py
```

 
## Troubles

[line 55](https://github.com/xu-song/tensor2tensor-eager/blob/master/train_wmt_eager.py#L55)

```py
logits, losses_dict = model(features)
print(model.variables)  # get nothing from model.variables
```
