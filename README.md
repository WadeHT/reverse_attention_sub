# reverse_attention_sub




| arguement     | value |
| --------      | -------- |
| dataset_root  | "sim_dataset_root": "~/dataset folder"    |
| save_model    | "~/model weight where want to save"       |
| save_img      | "~/imgs where want to save"               |
| save_log      | "~/log where want to save"                |
| weight_pth    | "~/weight where saved" / None             |
| max_epoch     | constant                                  |
| CUDA          | GPU number / None                         |


---

example:
train(dataset_root={"sim_dataset_root": "~/dataset folder"},
      save_model="~/model weight where want to save",
      save_img="~/imgs where want to save",
      save_log="~/log where want to save",
      weight_pth="~/weight where want to save" # if no init, weight_pth = None
      max_epoch=100,
      CUDA=0)
      
      



