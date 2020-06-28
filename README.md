NB: Here we assume that readers of this section are aleady familiar with neural networks fundamentals with respect to activation functions, forward and backpropagation techniques. 
If it is not the case don't worry! we have prepared some awsome references for you to get started :)

# Sequence-to-Sequnce neural networks architectures
## Introduction 

In order to deal with series of structured/unstructured data, we need to capture each granular sequence within the data serie. Most importantly, we need to keep track of the previous/next sequences. This is because, within the data serie, we know that each sequence is likely dependent on one or many other sequence(s). Hence it is crucial to accurately reflect this kind of dependencies relationship.

Recurrent neural networks (RNNs) are designed to handle sequence to sequence data, where each sequence can keep track of its backward peers. In some situations we need to go further and collect information on the forwad peers, whitch is also possible under bi-directional RNN models. In this post we will go through the model design of both uni/bidirectional RNNs and then we will explore some other RNNs variations with respect to additional enhancements and capabilities such as memory and attention.

## Practical usages
Areas such as natural language processing (NLP) and speech recognition are very good candidates to RNN architectures. You'll find bellow a non-exhaustive list of some comprehensive usages with real added value to various vital industries.


   | Usage | Input (X) | Output(Y) |
   | :--- | :--- | :--- |
   Machine translation | Sequence of words | Sequence of words |
   Sentiment classification | Sequence of words | Score |
   DNA analyis | Alphabet sequence | Protein mapping |
   Music generation | Set of notes as an integer | Sequence of music |
   Voice recognition | Audio clip | Sequence of words |
   Video recognition | Sequence of video frames | Entity, activity recognition|

Most of these situations are adressed through a supervised learning problem, ie given a representation of labeled data (X, Y), the RNN will use a sequence model to learn the adequate set of parameters in order to map a particular input X to the target output Y. 

In this post we will focus on NLP techniques to illustrate how RNNs can be used to handle text content where inputs and outputs can be different lenghts across different examples. We will also see how RNNs can allow features sharing across diffrent positions of the text (elements learned from one piece of a text, can generalize to the subequent parts of same text).

## Representation and notations 

First we need to care about the words representation (inputs and/or outputs)

- Step1: Build a sizable dictionnary (large corpus) that best reflects the words and terms universe within the area of interest. All words within the dictionnary are ordered and indexed.

- Step2: Apply word stemming if appropriate to your purpose (the key benefit here is the corpus size reduction)

- Step3: Tokenize your input. Each tokenized word is represented by a (one-hot) vector of same size as the dictionary. Set 1 at the position where the word matches the dictionary entry and 0 eslewhere. This is simply a binary projection of the word on the dictionnay space.

- Step4: Each (one-hot) vector from step2 is represented within a word embending space (dimentionality reduction techniques). 

| Notation | Description |
| :--- | :--- |
| (X,Y) | Pair X and Y of labeled data from the training set examples |
| Y^ | Predicted element given an input X |
| T<sub>x</sub><sup>(i)</sup> | The length of the sequence in the i<sup>th</sup> training example input |
| T<sub>y</sub><sup>(i)</sup> | The length of the sequence in the i<sup>th</sup> training example output|
| X<sup>(i)\<t\></sup> | The t<sup>th</sup> element/sequence in the i<sup>th</sup> training example input|
| Y<sup>(i)\<t\></sup> |The t<sup>th</sup> element/sequence in the i<sup>th</sup> training example output|
   

## Unidirectional recursive neural networks (RNN)

The figure bellow, illustrates a classic architecture for recurrent neural network. Here we would like to spot two main benefits drawn from this architecture :

1. X and Y can be of different size
2. Features cross-sharing, ie features learnt from earlier layers are shared and factored across later layers leading to a much more efficient way of learning model parameters 

<p align="center">
   <img  src="./rnn.png" alt="RNN!" title="Recurrent neural network (RNN)">
</p>

Observations:

- a<sup>\<0\></sup> can be initialized randomly, however common choice would be a zero vertor intialization;
   
- We can see that to predict the output Y^<sup>\<2\></sup>, the second layer takes into account the second word X<sup>\<2\></sup> as well as the activation value a<sup>\<1\></sup> from time 1. That is when making the prediction for X<sup>\<2\></sup> the RNN compiles information not only from  X<sup>\<2\></sup> but also from  X<sup>\<1\></sup>;
   
- The input parameters W<sub>ax</sub> are the same across the different layers. Similarely the activation and output parameters (resp.) W<sub>aa</sub> and W<sub>ay</sub> are shared across the different time steps;

- Broadly speaking speaking, the RNN processes input data from left to right. At each time step t, the RNN takes the input X<sup>\<t\></sup> and passes on the activation value a<sup>\<t\></sup> to the next step t+1;

- The activation values are calculated through a forward propagation process governed by the following equations:
   - a<sup>\<t\></sup> = g<sub>1</sub>(W<sub>aa</sub>a<sup>\<t-1\></sup> + W<sub>ax</sub>X<sup>\<t\></sup>  + b<sub>a</sub>);
   - Y^<sup>\<t\></sup> = g<sub>2</sub>(W<sub>ay</sub>a<sup>\<t\></sup> + b<sub>y</sub>);
   - The g<sub>1</sub> and g<sub>2</sub> may differ from each other. Generally we use functions such as tanh, ReLU and sigmoid.

- The parameters   W<sub>aa</sub>,  W<sub>ax</sub>, W<sub>ay</sub>, b<sub>a</sub>, b<sub>y</sub> are learnt through an optimizer such as gradient descent or equivalent, that minimizes the logistic loss function L(Y^,Y) stated bellow. This is an iterative process where the parameters are updated using the partial derivatives of L with respect to each parameter. The partial derivatives are obtained by runing a backward propagation procedure to train your RNN. 

   - L(Y^,Y) = - Sum<sub>t=1</sub><sup>T<sub>y</sub></sup>(Y<sup>\<t\></sup>log(Y^<sup>\<t\></sup>) + (1-Y<sup>\<t\></sup>)log(1-Y^<sup>\<t\></sup>))

- The architecture illustrated above has a many-to-many structure where inputs and outputs have equal length, that is (T<sub>x</sub> = T<sub>y</sub>). However in the case of speech recognition, a one-to-many architecture is more suitable. Similarly a many-to-one architecture will fit better with sentiment analysis needs. Finally the encoder-decoder architecture (many-to-many with T<sub>x</sub> different from T<sub>y</sub>) works well for mahine translation purposes.

- One drawback of unidirectional RNNs is that for a given time t, it does not look ahead for information that comes later in the sequence. It only incorporates previous information from earlier time steps. 

In particular if you need to get the full picture of a data serie before predicting the result for each time t, then the Bidirectional recursive neural networks (BRNN) architecture is more suitable.

## Bidirectional recursive neural networks (BRNN)

## Gated Recurrent Units (GRU) and Long short term model (LSTM)
The basic RNN model suffers from lack of long term memory due to vanishing gradients problem, in particular when the data squence exhibits long term dependencies. That is elements from later time steps are dependent on elments from the very early time steps. 
In fact, as the RNN get deeper, the gradients from latest layers experience a hard time propagating back to update the paramters of the very early layers. Please note that exploding gradients may lead to the same situation in whitch case the gradient clipping may be used as a workaround solution to fix the problem.

The Gated Recurrent Units (GRU) model is a variation of the RNN model, it has been designed to adress the lack of long term memory also known as "local influences problem". Technically speaking the GRU adds a new output for each layer unit called the "memory cell", denoted as c<sup>\<t\></sup>. c<sup>\<t\></sup> will carry out any desired activation value from earlier layers up to the time step t where it is no more needed and will be replaced by the candidate cell memory value c-tilda<sup>\<t\></sup>. c<sup>\<t\></sup> is driven by an update-gate sigmoid function G<sub>u</sub> that takes 1 when the cell memory needs to be replaced by c-tilda<sup>\<t\></sup> and 0 otherwise. c-tilda<sup>\<t\></sup> is also driver by a relevance gate. that is how much c<sup>\<t-1\></sup> is relevant to the calculation of the candidate c-tilda<sup>\<t\></sup>.
   
   - c<sup>\<t-1\></sup> = a<sup>\<t-1\></sup>
   
   - G<sup>\<r\></sup> = sigmoid(W<sub>rc</sub>c<sup>\<t-1\></sup> + W<sub>rx</sub>X<sup>\<t\></sup>  + b<sub>r</sub>)
   
   - c-tilda<sup>\<t\></sup> = tanh( G<sub>\<r\></sub>W<sub>cc</sub>c<sup>\<t-1\></sup> + W<sub>cx</sub>X<sup>\<t\></sup>  + b<sub>c</sub>)
   
   
   - G<sup>\<u\></sup> = sigmoid(W<sub>uc</sub>c<sup>\<t-1\></sup> + W<sub>ux</sub>X<sup>\<t\></sup>  + b<sub>u</sub>)
   
   - c<sup>\<t\></sup> = G<sub>\<u\></sub> c-tilda<sup>\<t\></sup> + (1-G<sub>\<u\></sub>) c<sup>\<t-1\></sup>

## Attention model

## Conclusion

