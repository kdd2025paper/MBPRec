# MBPRec

This repository contains the implementation of the paper:

## Parameter Settings

We optimize the approach using the Adam optimizer with a learning rate of `0.001`.

- **Training Batch Size**: Chosen from `[1024, 2048, 4096, 6114]`.
- **Embedding Dimension**: Selected from `[64, 128, 256, 512]`.
- **Task Weight Parameter (λ)**: Tuned within `[0.05, 0.1, 0.2, 0.5]`.
- **Matching Coefficient (lpop)**: Picked from `[0.05, 0.1, 0.2, 0.5, 1.0]`.
- **Temperature Coefficient (τ)**: Explored within `[0.1, 0.2, 0.5, 1.0]`.

## Example to Run the Code

To train and evaluate the model, use the following command:

```bash
python main.py
```

## Suggestions for Parameters

Two important parameters need to be tuned for different datasets:

- `self.lamda = [0.05, 0.1, 0.2, 0.5]`
- `self.mu = [0.05, 0.8, 0.1]`

Specifically, we suggest tuning `lamda` among `[0.1, 0.012, 0.15, 0.17, 0.2]`. Generally, this parameter is related to the sparsity of the dataset. If the dataset is more sparse, a smaller value of `lamda` may lead to better performance.

## Statement

For some reasons, we do not recognize **MBPRec** (Multi-behavior Popularity-aware Recommendation) as a state-of-the-art method for multi-behavior recommendation. We also call on researchers to not only compare with MBPRec in future research, so as to avoid getting an inaccurate conclusion.