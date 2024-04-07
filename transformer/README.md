# Introduction

In this project, I will build the transformer from scratch by **torch**, and use some simple experiments to test the validation of my transformer.

## Details of Transformer
Becase the structures and details have been explored a lot, I will list the implementation details of my transformer here to avoid misunderstand.
- Positional encoding:
    1. I use the original, simple and untrainable encoding: **sinusoidal positional encoding**.
- Self-attention:
    1. I discord the original order of residual net and attention and use **'pre-norm'**, which means we use residual method outside of the layernorm, as for the better performance.
    2. I will implement the **'distance-enhanced attention scores'** method and check the its performance by ablation study.

### Model hyper-parameters
- vocab_size: size of vocabulary (including special characters)
- embedding_d: dimension of embedding
- head_n : head nmber for self-attention
- layer_n: number of layers for encoder and decoder

## Experiments
All experiments are finished in jupyter notebook in order to convinently save the middle results and check the validation.

- **test experiment**: check the implementation of MyTransformer and mainly check the dimension of middle process.

- **experiment 1**: equence Reversal: reversing a sequences of numbers, [1,3,2,4,5,3] -> [3,5,4,2,3,1]
    1. In this experiment, every ouput only needs to focus on one input number and output order. Therefore, it could only require lower context window than the whole context. 
    2. We will try to use 'distance-enhanced' attention socres to learn this task quicker.

- **experiment 2**: equence Reversal for negative number and big numbers: reversing a sequences of numbers, [1,-3,2,435,5,3] -> [3,5,435,2,-3,1].
    1. Compared to the first experiment, the transformer need to recognize the minus '-' and the big number, so there is a harder version and we hope to implement it by adding the special word <delimiter>