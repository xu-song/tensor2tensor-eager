
# t2t_trainer in eager model



## Run



```bash
git clone https://github.com/xu-song/tensor2tensor-eager.git
cd tensor2tensor-eager
python train_wmt_eager.py
```

 
## Troubles

```py
logits, losses_dict = model(features)
print(model.variables)  # get nothing from model.variables
```
