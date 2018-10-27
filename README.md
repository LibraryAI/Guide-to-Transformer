# About Generative Pre-Training, and Transformer
Attention 계열 논문 중 question answering 과 맥락 추측 등 natrual language understanding 분야에서 SOTA 퍼포먼스를 보이는 ['Improving Language Understanding by Generative Pre-Training'](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)을 제대로 살펴보고자 한다.<br>
<p align="center">
<img src="https://cdn-images-1.medium.com/max/1000/1*aC5E6WcoOX8mZ4MHw72zjg.png" alt="Drawing" style="width: 200px;" title="General structure of decoer only transformer"/>
</p>
<br>
  
위 논문에서는 RNN 등 autoregressive 모델을 사용하지 않은 Feed Forward Network와 Attention mechanism만을 이용한 'Attention is all you need!' 논문의 Transformer 모델을 이용했다. Transformer와의 차이점은 Transformer의 encoder-decoder 구조와 다른 decoder only 모델을 사용한 점과 task-specific한 작업에 대해 unsupervised pre-training을 통해 학습한 랭귀지 모델을 인풋과 final layer 변형을 통해서 효율적으로 학습할 수 있다는 차이가 있다.

## Unsupervised Pre-training
이 작업을 통해서 논문에서 기술한 byte-pair encoding, positional embedding, masked multi-headed self attention 등을 살펴보고자 한다. <br></br>

### Part1. Byte-Pair Encoding
- __Create BPE pair set__: 데이터셋에서, byte-pair encoding pair set 을 생성<br>
  - Hyperparameter : 40,000 ; 40,000 개의 pair 유지<br>
  - [자세한 개념과 코드 설명](https://github.com/LibraryAI/NLP/blob/master/make_BPE.ipynb) <br></br>

### Part2. Text and Position Embedding
- __Apply BPE pair tokenization__
  - Batch data에 BPE 를 수행해 BPE pair Tokenization 
  - [자세한 개념과 코드 설명](https://github.com/LibraryAI/NLP/blob/master/apply_BPE.ipynb) <br>
- __Positional encoding and Word embedding__
  - Fixed Sine, Cosine embedding func applied without training. Same dimension as the embedding (256) and is added to the embedding. 벡터 합. 사이즈 변화 X
  - BPE Tokenized Data를 Word Embedding
  - Hyperparameter : 256 ; 256 dim embedding vector space 
  - [자세한 개념과 코드 설명](https://github.com/LibraryAI/NLP/blob/master/about_WE_and_PE.ipynb)<br></br>
  
### Part3. Multilayer Transformer Decoder
- __Self attention__
  - __Key/Value/Query matrix__
    - Batch embedding data(as X) 가 MbyN matrix라면, 세 개의 독립된 NbyT weight matrix를 곱해 MbyT matrix 세개를 구함
    - M: Batch size, N: embedding dim, T: New vector dim
    - Hyperparameter: M: , N:256, T:64
  - __Score__
    - Dot product Query and Key Matrix, (Q×Transpose(K)), for each word embedding Q1×T(K1), Q1×T(K2), Q1×(K3)....
  - __Divide and Softmax__
    - Divide by sqrt(dim of T = 64) = 8 to stabilize gradient descent. 한 word embedding의 score/8 값들에 softmax
  - __Weighted Values__
    - Multiply Value matrix by Softmax outcome, outcome is the weighted value of words in the sentence by the specific word
  - __Attention__
    - Sum all the weighted values (as Z)
  - [자세한 개념과 코드 설명](https://github.com/LibraryAI/NLP/blob/master/about_transformer_self_attention.ipynb)
- __Multi-headed self attention__
  - __Multiple matrices__
    - Create multiple number of different self attention matrices
    - Hyperparameter: 12
  - __Concatenation__
    - Concatenate Z1, Z2, Z3, .. Zn in row. (Z: M by n*T size matrix)
  - __Weight All__
    - Multiply concatenated Z with Weight matrix W of n*T by N, Z × W = Z is back again with M by N size
  - [자세한 개념과 코드 설명](https://github.com/LibraryAI/NLP/blob/master/about_transformer_self_attention.ipynb)
- __Masked multi-headed self attention__
  - __Masking future time step scores__
    - Self attention의 세번째 step의 softmax를 적용하기 전에 해당 word의 future time step 에 해당하는 word vector들의 score -inf 로 바꿔주기
    - 다른 부분은 self attention과 동일
- __Normalization Layer__
  - __Residual Connection__
    - attention 레이어의 결과값인 Z 와 Positional embedding 매트릭스 X 를 더해준다 (Same size)
  - __Normalization__
- __Position-Wise Feed-Forward Net__
    - FFN(x) = max(0, xW1 + b1)W2 + b2 (using RELU)
    - W1: 256 by 1024, W2: 2048 by 256, b1: 1 by 1024, b2: 1 by 256
- __Normalization Layer__
  - Residual Connection
    - FFN 의 결과값과 4번 Normalization layer의 결과값을 더해준다
  - Normalization
### Part4. Final Layer
- __Linear__
  - transforms the matrix into dim(vocabulary)
- __Softmax__
  - 각 단어의 likelihood를 계산
- __Log likelihood __

