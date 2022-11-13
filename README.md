# FastFormer

## A Tensorflow Implementation of Fastformer (Fast Transformer) 

This is a TensorFlow implementation of [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084) by Wu et al. 

**Fastformer** is a Transformer variant that uses _Additive Attention_, a new type of attention based on additive operation rather than multiplicative. This circumvents the quadratic bottleneck that we usually have with attention making it much more efficient. It can achieve comparable or even better results for long sequence modeling performance.

![architecture.png](images/architecture.png)


## License

```
   Copyright 2022 Alaa Awad

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