dataset_names = ["cifar10"]#"mnist"（表现良好，先放一放）, "cifar10"，


# 调参，一次实验一个模型调一个参数
model_configs = {
    "MLP": {
        "epochs": [10],  
        "learning_rates": [8e-5],
        "hidden_sizes": [128],
        "batch_sizes": [8]     
    },
    "CNN": {
        "epochs": [30],
        "learning_rates": [0.0015],
        "batch_sizes": [8]
    },
    "ViT": {
        "epochs": [30],
        "learning_rates": [0.0015],
        "batch_sizes": [8]
    }
}