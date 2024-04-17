# TokenLearner

![](https://github.com/google-research/scenic/raw/main/scenic/projects/token_learner/data/tokenlearner.gif)

PyTorch implementation of TokenLearner. TokenLearner is a learnable module to be placed within Transformer architectures for images and videos. Once placed, it significantly reduces the number of tokens for all subsequent layers, thereby reducing the overall computation. It simultaneously increases accuracy of the models by making the tokens dynamic and adaptive to the input.

## Usage

### TokenLearner
This is the implementation of TokenLearner introduced in the paper: https://arxiv.org/abs/2106.11297

For 1D input (with `[B, N, C]` shape):
```python
from model import TokenLearner, TokenFuser

num_tokens, in_channels = 8, 256
tokenlearner = TokenLearner(num_tokens, in_channels)
tokenfuser = TokenFuser(num_tokens, in_channels)

inputs = torch.zeros(1, 16, in_channels)
out = tokenlearner(inputs)
fused = tokenfuser(out, inputs)

print(inputs.shape, out.shape, fused.shape) 
# torch.Size([1, 16, 256]) torch.Size([1, 8, 256]) torch.Size([1, 16, 256])
```

For 2D input (with `[B, C, H, W]` shape):
```python
inputs = torch.zeros(1, in_channels, 4, 4)
out = tokenlearner(inputs)
fused = tokenfuser(out, inputs)

print(inputs.shape, out.shape, fused.shape) 
# torch.Size([1, 256, 4, 4]) torch.Size([1, 8, 256]) torch.Size([1, 256, 4, 4])
```

### TokenLearnerV11

TokenLearner module Version 1.1, using slightly different conv. layers. 

> Instead of using 4 conv. layers with small channels to implement spatial attention, this version uses a MLP with gelu inbetween. It also uses softmax instead of sigmoid. **This version works better in general.**

For 1D input (with `[B, N, C]` shape):
```python
from model import TokenLearnerV11, TokenFuser

num_tokens, in_channels = 8, 256
tokenlearner = TokenLearnerV11(num_tokens, in_channels)
tokenfuser = TokenFuser(num_tokens, in_channels)

inputs = torch.zeros(1, 16, in_channels)
out = tokenlearner(inputs)
fused = tokenfuser(out, inputs)

print(inputs.shape, out.shape, fused.shape) 
# torch.Size([1, 16, 256]) torch.Size([1, 8, 256]) torch.Size([1, 16, 256])
```

For 2D input (with `[B, C, H, W]` shape):
```python
inputs = torch.zeros(1, in_channels, 4, 4)
out = tokenlearner(inputs)
fused = tokenfuser(out, inputs)

print(inputs.shape, out.shape, fused.shape) 
# torch.Size([1, 256, 4, 4]) torch.Size([1, 8, 256]) torch.Size([1, 256, 4, 4])
```

## Acknowledgement

* [TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner): official JAX implementation of TokenLearner.


## License
```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
