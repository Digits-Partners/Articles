NB: Here we assume that readers are aleady familiar with neural networks fundamentals with respect to activation functions, forward and backpropagation techniques. 
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

- Step4: Each (one-hot) vector from step3 is represented within a word embending space (dimentionality reduction techniques). 

| Notation | Description |
| :--- | :--- |
| (X,Y) | Pair X and Y of labeled data from the training set examples |
| Y^ | Predicted element given an input X |
| T<sub>x</sub><sup>(i)</sup> | The length of the sequence in the i<sup>th</sup> training example input |
| T<sub>y</sub><sup>(i)</sup> | The length of the sequence in the i<sup>th</sup> training example output|
| X<sup>(i)\<t\></sup> | The t<sup>th</sup> element/sequence in the i<sup>th</sup> training example input|
| Y<sup>(i)\<t\></sup> |The t<sup>th</sup> element/sequence in the i<sup>th</sup> training example output|
   
## Words embeding (WE)

There is some analogy between image encoding through convolutional nets (resulting in a full connected layer), and words embeding through a neural network that learns the embeding vectors given a large text corpus. Yet there is one slight difference between the two, that is in the case of word embeding, the text corpus is fixed and embeded vectors are learnt only for that text corpus, whereas in the image encoding, training set is dynamic, ie the neural network learns encoding even for new images.

(WE) captures the relationship between words such as analogies, similarities, or any other business logic that might be of interest. Here some of the key apsects of this concept:

1. Based on the words corpus, identify/define  a space of features, called embeding space, that would best capture the relationships between words wihtin your domain of interest. [Mikolov et. al., 2013, Linguistic regularities in continous space word representation].

2. Construct the embedding matrix (number of embedding vectors x number of words in the text corpus), to represent each word in the new embeding space by electing one of the following options:

   1. Learn a featurized representation from a large text corpus (up to hundred of billions of words), works well in machine translation. Some of the most popular algorithms are decribed bellow:
   
      1. [Bengio et. al,2003, A neural probabilistic language model], predicts a target word (t) given a certain context of previous words (c). This can done by training a one hidden layer neural network, that takes as inputs the one-hot representation of the context (c), and feeds the embedding vectors associated with that context to a softmax unit whitch in turn classifies the given the embedded context among all possible words within the text corpus, such as:
      
            - Softmax prediction : Y^ = P(t/c) = (e<sup>θ<sub>t</sub><sup>T</sup>e<sub>c</sub></sup>) / Sum<sub>j=1</sub> <sup>n</sup>(e<sup>θ<sub>j</sub><sup>T</sup>e<sub>c</sub></sup>), where:

              - θ<sub>t</sub> : the target weights (as model parameters of the softmax unit);

              - e<sub>c</sub> : the context embedding vector (as model parameters of the hidden layer);

              - n : the corpus text size

            - Loss function : L(Y, Y^) = -Sum<sub>i=1</sub><sup>n</sup>(Y<sub>i</sub>log(Y^<sub>i</sub>))
      
         To learn the model parameters (the targets weights θ<sub>t</sub> and the context embedding vectors e<sub>c</sub>), we backpropagate loss partial derivatives with respect to these parameters, then we run the loss optimization process (gradient descent or equivalent) to maximize the training set log-likelihood. 
         
         One extension of the above algorithm is to predict a target word given a certain context of words arround the target.         
      
      2. The Skip-grams algorithm is also another extension of the algorithm descirbed above. Indeed it maps a context to a target word, where the target is within a window of n-words nearby the context. That is given a context of words, we may skip few (previous/next) words to reach out the target. [Mikolov et. al., 2013. Efficient estimation of word epresentation in vector space]
      
      3. In the case of large scale neural networks, some computational challenges may araise while using the softmax classifier. Making prediction slows down as the text corpus gets larger. This is because for each prediction we need to sum up over all the embedding vectors of text corpus. In this case, the computaitonal cost scales linearly with the text corpus size. 
      
         One way of speeding up the softmax classifier is to use a binary classifier called hierarchical softmax. In this configuration the text corpus is represented by a binary tree, namely Huffman tree, where the root node (parent) holds the complete text corpus, the leafs can be represented by the individual words in the text corpus. Each parent node is subdivided into two distinct sub groups (parent = union of sub groups). 
         
         The tree is not balanced, the nodes are of different sizes. The idea here is to choose a heuristic that minimizes the path length from root to leaf, in particular for frequent words, pushing down infrequent words deeper in the tree. In addition, when sampling the context, excessively frequent words that are irrelevant to the words embedding exercise, should be ignored/assigned very low weights by the chosen heuristic.
                  
         The hierarchical softmax starts from the root node down to the leaf following a decision process, where the transition probability (conditional to context), to go from parent to child on the right vs. left, is driven by the normalized sum of the underlying probabilities of the children. Untimately the probability of leafs corresponds to the distribution function of the words in the corpus text. In the case of hierarchical softmax, the computaiton cost scales by log of the text corpus size.
         
         Finally, the computational cost can be further reduced through negative sampling. [Mikolov et. al., 2013. Distributed representation of words and phrases and their compositionality]
            
            - Rather than running a T<sub>x</sub> softmax classifier (one shot training set classification among T<sub>x</sub> words), we run T<sub>x</sub> distinct logistic regressions on much smaller training sets comprised of one positive example and k negative examples. Negative examples are randomaly sampled from the text corpus, where k range [2-5] for large datasets and [5-20] for smaller datasets.
             
            - The logistic regression model takes as inputs k+1 pairs of (context, target) and trains a shallow neural network to learn the parameters (θ<sub>t</sub> and e<sub>c</sub>) and predict Y^, either to be positive or negative for each word in the text corpus, ie we run T<sub>x</sub> binary classfiers on a training set of size k+1, where k << T<sub>x</sub>, hence it is computationaly less expensive than the softmax classifier whitch performs the classification on a much larger training set.
        
              - Sigmoid prediction : Y^ = P(Y=1/t,c) = 1/(1+e<sup>-θ<sub>t</sub><sup>T</sup>e<sub>c</sub></sup>)
              - θ<sub>t</sub> : the target weights (as model parameters of the sigmoid unit)
         
              - e<sub>c</sub> : the context embedding vector (as model parameters of the hidden layer)
  
            - the authors of the paper referred above, recommend to use a heuristic to sample Negative examples based on their frequency, yet they found a metric somewhere in between uniform distribution and empirical frequency distribution 
              - P((w<sub>i</sub>) = f(w<sub>i</sub>)<sup>3/4</sup> / Sum<sub>j=1</sub> <sup>T<sub>x</sub></sup>(f(w<sub>j</sub>)<sup>3/4</sup>), where f is empirical distribution function for words in the text corpus.

      3. GloVe algorithm (less popular than skip-gram or word2vec), but still interesting to consider [Pennington et. al., 2014. GloVe: global vectors for word representation]
      
            - Given a sampled pair of words, context (c) and target (t), let X<sub>tc</sub> be the number of times that a target word t appears in the context c, that is how frequent t appears close to the context c.
            - Learn embedding vectors e<sub>c</sub> and target vectors θ<sub>t</sub><sup>T</sup>, such as their innner product minimizes the square distance to the log frequency vector X<sub>tc</sub>. Gradient descent can be used to optimize this square distance:
            
            - minimize the loss function: Sum<sub>t=1</sub> <sup>T<sub>x</sub></sup> (Sum<sub>c=1</sub> <sup>T<sub>x</sub></sup>f(X<sub>tc</sub>)(θ<sub>t</sub><sup>T</sup>e<sub>c</sub> + b<sub>t</sub> + b<sub>c</sub> - logX<sub>tc</sub>)<sup>2</sup>), where f is a weighting term that neutralize the case where X<sub>tc</sub>=0. In addition the choice of this heuristic f, should prevent from giving less frequent words too little weights and giving frequent words too much undue weights.
            
            - to train the algorithm, initialize e<sub>c</sub> and θ<sub>t</sub> with uniformally random distibution, run gradient descent to minimize the loss function, and then for each word, take the average of e<sub>c</sub> and θ<sub>t</sub> to compute the final embedding vector for a given pair (c,t). Here we can simply use the arithmetic average because e<sub>c</sub> and θ<sub>t</sub> are symetric under the GloVe model.

      4. One last observation regarding all the algorithms mentioned above, is that they can not guarantee that the embedding vectors learnt, are always interpretable as θ<sub>t</sub><sup>T</sup>e<sub>c</sub> is not always aligned with features axis that humans can easily comprehend.
      
   2. Take advantage from an existing pre-trained word embeding network and transfer its learning to your specific task (smaller training set), in particular tasks like name entity recognition, core reference resolution, text sumurization
   
 3. Similarities that hold in n-densional space, may not hold after t-SNE mapping. t-SNE algorithm takes an n-dimensional vector and maps it in a non-linear way to a 2-dimensional space

 4. One popular way of implementing the similarity function between two words -represented respectively by their associated embeding vectors u and v- is based on the cosine of the angle between the two emebeding vectors u and v, and is expressed as follows:
 
    - cosine(u,v) = (u<sup>T</sup>v) / (||u||<sub>2</sub> ||v||<sub>2</sub>)
    
    Another way of measuring the similarity between u and v, is based on the euclidean distance between these two vectors.

 3. Sentiment classification problem allows to score a piece of text to serve a decision-taking process. Although there is a lack of sizable training set, words embedding helps to train algorithms even on small datasets

    - Transfer learnt embedding vectors from pre-trained algorithm and use them to represent words in the text corpus
    
    - Given a piece of text, extract the embedding vectors for each word (multiply the word one hot vector by the embedding matrix) and average the resulting embedding vectors that represent the piece of text being scored.
    
    - Feed the average embeding vector to a softmax classifier to predict the score.
    
    - This construction is orthogonal to words order. for instance if positive words  get frequently involved in a negative sentiment, the softmax classifer will predict a false good score. One way of solving this issue is to use RNNs architectures.
    
    - Train a many-to-one RNN that takes as inputs the embedding vectors associated to the words from the piece of text being scored, and feed the last forward activation value to a softmax classifier to predict the score.
    
 5. Reduce/neutralize undesirable biases from text corpus, such as gender, ethnicity, relegion, wealth and other biases, [Bolukbasi et. al., 2016. Man is to computer programmer as woman is to homeworker? Debiasing word embeddings].

    - Identify the bias axis and its orthogonal axis (non-bias). The bias axis can be derived by reducing the dimentionaly of the emebedding vectors through SVD (PCA like algorithm). if n is the size of the embedding vectors and m the nummber of principal components, then bias axis dimension equals m and the non-bias diemnsion equals n-m.

    - Train a linear classifier to identify whitch words are specific to a particular bias and those that are not.

    - Neutralize the bias for words that are not instrinsically/legitimately related to that bias. This can be done by projecting these words on the non-biais axis
    
    - re-align the intrinsic pairs on the non-bias axis, to ensure that a non-intrinsic word stands at the same distance from the intrinsic words pair.
    

 
 
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

- The architecture illustrated above has a many-to-many structure where inputs and outputs have equal length, that is (T<sub>x</sub> = T<sub>y</sub>). However in the case of speech recognition, a one-to-many architecture is more suitable. Similarly a many-to-one architecture is a more appropriate fit when it comes to sentiment analysis. Finally the encoder-decoder architecture (many-to-many with T<sub>x</sub> different from T<sub>y</sub>) works well for mahine translation purposes.

- Unidirectional RNNs arichitecture put some training challenges, in particular when:
   1. The RNN gets deeper, the vanishing/exploding gradients prevent from properly learning the model parameters. The GRU and LSTM sections describe how this issue has been resolved.   
   2. Your prediction at time step t, depends on information that comes later in the sequence. Classic RNNs only incorporates information prior to t. In order to capture information that comes later on, ie after time step t, the Bidirectional recursive neural networks (BRNN) architecture is more convinient choice.


## Gated Recurrent Units (GRU)
The basic RNN model suffers from lack of long term memory due to vanishing gradients problem, in particular when the data squence exhibits long term dependencies, where elements from later time steps are dependent on elments from the very early time steps. 
In fact, as the RNN gets deeper, the gradients from latest layers struggle in propagating back to update the paramters from the very early layers. Exploding gradients may lead to the same situation, however the gradient clipping may be used as a workaround solution to fix the problem.

- The Gated Recurrent Units (GRU) model is a variation of the RNN model, it has been designed to adress the lack of long term memory also known as "local influences problem". Technically speaking the GRU create a new output for each layer unit, called the "memory cell", denoted as c<sup>\<t\></sup>. 
- c<sup>\<t\></sup> carries out any desired activation value from earlier layers up to the time step t where it is no longer needed. It will then be replaced by the new candidate cell memory value c-tilda<sup>\<t\></sup>. 

- The memory update decision is driven by an update-gate sigmoid function G<sub>u</sub> that takes 1 when the cell memory needs to be replaced by c-tilda<sup>\<t\></sup> and 0 otherwise. 

- The candidate cell memory c-tilda<sup>\<t\></sup> is also driven by a relevance gate, denoted G<sub>r</sub>, indicating how much c<sup>\<t-1\></sup> is relevant to the calculation of the candidate c-tilda<sup>\<t\></sup>.
   
   - c<sup>\<t-1\></sup> = a<sup>\<t-1\></sup>
   
   - G<sub>r</sub> = sigmoid(W<sub>rc</sub>c<sup>\<t-1\></sup> + W<sub>rx</sub>X<sup>\<t\></sup>  + b<sub>r</sub>)
   
   - c-tilda<sup>\<t\></sup> = tanh( G<sub>r</sub>*W<sub>cc</sub>c<sup>\<t-1\></sup> + W<sub>cx</sub>X<sup>\<t\></sup>  + b<sub>c</sub>)
   
   
   - G<sub>u</sub> = sigmoid(W<sub>uc</sub>c<sup>\<t-1\></sup> + W<sub>ux</sub>X<sup>\<t\></sup>  + b<sub>u</sub>)
   
   - c<sup>\<t\></sup> = G<sub>u</sub>*c-tilda<sup>\<t\></sup> + (1-G<sub>\<u\></sub>)*c<sup>\<t-1\></sup>

Under such a construction, even when G<sub>u</sub> gets very small (due vanishing gradients), c<sup>\<t\></sup> will keep track of the memorised value c<sup>\<t-1\></sup>.

## Long short term model (LSTM)
LSTM stands from Long term short memory, it is a more general version of GRU, with the folowing variations:

   - c-tilda<sup>\<t\></sup> = tanh( W<sub>ca</sub>a<sup>\<t-1\></sup> + W<sub>cx</sub>X<sup>\<t\></sup>  + b<sub>c</sub>), (no relevance gate G<sub>r</sub>. uses a<sup>\<t-1\></sup> rather than c<sup>\<t-1\></sup>) 
   
   - G<sub>u</sub> = sigmoid(W<sub>uc</sub>c<sup>\<t-1\></sup> + W<sub>ux</sub>X<sup>\<t\></sup>  + b<sub>u</sub>), (This the update gate, same as the one used in the GRU model)
 
  - G<sub>f</sub> = sigmoid(W<sub>fa</sub>a<sup>\<t-1\></sup> + W<sub>fx</sub>X<sup>\<t\></sup>  + b<sub>f </sub>), (The forget gate G<sub>f</sub> replaces the (1-G<sub>u</sub>) term in the GRU model, used to carry the old memory cell value c<sup>\<t-1\></sup>)
  
  - G<sub>o</sub> = sigmoid(W<sub>oa</sub>a<sup>\<t-1\></sup> + W<sub>ox</sub>X<sup>\<t\></sup>  + b<sub>o </sub>), (This is the output gate, it drives the values of a<sup>\<t\></sup> given c<sup>\<t\></sup>)
  
  - c<sup>\<t\></sup> = G<sub>u</sub>*c-tilda<sup>\<t\></sup> + G<sub>f</sub>*c<sup>\<t-1\></sup>
 
  - a<sup>\<t\></sup> = G<sub>o</sub>*tanh(c<sup>\<t\></sup>)
 
(*), denotes elements wise product

The forget gate G<sub>f</sub>, allows to store in the old memory cell value c<sup>\<t-1\></sup>. By construction the LSTM is more complex/flexible than GRUs, but they require more ressources to train the model parameters.
 
## Bidirectional recursive neural networks (BRNN)
The main benefit from BRNNs is that for a given time step t, it captures information from earlier and later sequences, ie sequences placed before and after t.   

The BRNN is defined by an acyclic graph, where given an input sequence X<sup>\<1\></sup> to X<sup>\<T<sub>x</sub>\></sup>, the neural networks runs the forward propagation through two levels of activation:
1. A forward sequence of activation values starting from fwd_a<sup>\<1\></sup> to fwd_a<sup>\<T<sub>x</sub>\></sup> 
2. A backward sequence of activation values starting from bwd_a<sup>\<T<sub>x</sub>\></sup> to bwd_a<sup>\<1\></sup>

The prediction Y^ at time t is based on both forward and backward activation values at time t (fwd_a<sup>\<t\></sup> and bwd_a<sup>\<t\></sup>), whitch allows the prediction to factor in information from steps prior to time t, as well as information after time t.
 
Elementary blocks within the BRNN can either be GRU or LSTM.

## Sequence-to sequence models
Mainly used in machine translation and speech recognition [Sutskever et. al., 2014. Sequence to sequence learning with neural networks] and [Cho et. al., 2014 Learning phrase representation using RNN encoder-decoder for statistical machine learning]

- Machine translation could be achiedved through an encoder-decoder architecture, built based on a many-to-one RNN (encoder either GRU or LSTM) that takes as inputs the original language sequence and feeds the last forward activation value to a one-to-many RNN (the decoder either GRU or LSTM) whitch outputs the translated sequence in the target language. 

This type of architecture works also well for image captioning, where the model predicts the textual expressions of the items present in an image. [MAo et. al., 2014. Deep captioning with multimodel recurrent neural networks] [Vinyals et. al., 2014. Show and tell: Neural image generator] [Karpathy and Li et. al., 2015. Deep visual-semantic alignments for generating image descriptions].

In this case the encoder will mostly use a pretrained convolutional networks (AlexNet, LeNet ...) to learn the features vector for the input image (up to the last full connected layer). The features vector then feeds to a one-to-many RNN (GRU or LSTM) to outputs the text sequence related to the items that could be identified on the input image.

Traditional language models estimate the conditional probability of a particular target word by sampling at random, the words from corpus distribution given the context word(s). 

in the case of Machine translation models, the decoder plays the role of a language model except it is now conditional to the encoder input X. In addition the sampled translated sentence, from the conditional distribution, should be at best fit and not at random, ie rather than just picking randomaly one translation sampled from the conditional distribution, we want to search for the best translation that maximizes the conditional probability. 

  - argmax <sub>Y<sup>\<1\></sup>, ..., Y<sup>\<T<sub>y</sub>\></sup></sub> P(Y<sup>\<1\></sup>, ..., Y<sup>\<T<sub>y</sub>\></sup> / X)

A couple of algorithms maight be used to solve this optimization problem:

  - Greedy search: it looks for the best first word that maximizes the conditional probability P(Y<sup>\<1\></sup> / X) and then looks for the second word that that maximizes P(Y<sup>\<1\></sup>, Y<sup>\<2\></sup> / X) and so forth up to P(Y<sup>\<1\></sup>, ...,Y<sup>\<T<sub>y</sub>\></sup> / X). This approach does not work well as it maximizes only the marginal cumulative probability of each predicted word but not the joint probability of all predicted words simulataneously.
 
 - Beam search : Given a hyper parameter B called "Beam width" that serves to select the B most likeley words, the Beam search approximizes the predicted translation by searching for Y that maximizes the joint conditional probability P(Y<sup>\<1\></sup>, ..., Y<sup>\<T<sub>y</sub>\></sup> / X). 
 
    1. To predict the first word, the beam search evaluates the distribution of P(Y<sup>\<1\></sup> / X) using a softmax classifier among  T<sub>x</sub> words in the text corpus, given X. Next the beam search will retain only the B most likely possible words, let's say B=3, that is words w11, w12, w13.
 
    2. For the second word prediction, the beam search uses a portion of the decoder network from Y<sup>\<1\></sup> to Y<sup>\<2\></sup> to evaluate three distributions (B=3), namely P(Y<sup>\<2\></sup> / X, Y<sup>\<1\></sup>=w11), P(Y<sup>\<2\></sup> / X, Y<sup>\<1\></sup>=w12) and P(Y<sup>\<2\></sup> / X, Y<sup>\<1\></sup>=w13). Next it will select the three pairs of words that maximizes the joint probability:
 
       P(Y<sup>\<1\></sup>, Y<sup>\<2\></sup> / X) =  P(Y<sup>\<1\></sup> / X) x P(Y<sup>\<2\></sup> / X, Y<sup>\<1\></sup>={w11, w12, w13}), where Y<sup>\<2\></sup> is ditributed among the T<sub>x</sub> words in the text corpus. 
 
       In this case the algorithm will pickup the most likely P(Y<sup>\<1\></sup>, Y<sup>\<2\></sup> / X) among 3xT<sub>x</sub> joint probabilities.
 
    3. The algorithm continues on processing the next words in the sequence, using the same procedure described in (2). When it reaches the last word to predict Y<sup>\<T<sub>y</sub>\></sup>, the beam search uses the full decoder network from Y<sup>\<1\></sup> to Y<sup>\<T<sub>y</sub>\></sup> to evaluate three distributions, namely:
       - P(Y<sup>\<T<sub>y</sub>\></sup> / X, {Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>}={uplet<sub>1</sub>}) 
       - P(Y<sup>\<T<sub>y</sub>\></sup> / X, {Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>}={uplet<sub>2</sub>})
       - P(Y<sup>\<T<sub>y</sub>\></sup> / X, {Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>}={uplet<sub>3xT<sub>x</sub></sub>}), where uplet1, uplet2, uplet<sub>T<sub>y</sub>-1</sub> represent each a (T<sub>y</sub>-1) sequence of predicted words. Finally the beam search selects the uplet of words that maximizes the joint probability:
   
       - P({Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>\></sup>}  / X) =  P({Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>} / X) x P(Y<sup>\<T<sub>y</sub>\></sup> / X, {Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>}=[{uplet<sub>1</sub>}, {uplet<sub>2</sub>},...,  {uplet<sub>3xT<sub>x</sub></sub>}]), 
       
       - P({Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>\></sup>}  / X) =  Product<sub>t=1</sub><sup>\<T<sub>y</sub>\></sup> P(Y<sup>\<t\></sup> / X, {Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>}=[{uplet<sub>1</sub>}, {uplet<sub>2</sub>},...,  {uplet<sub>3xT<sub>x</sub></sub>}])
 
         where Y<sup>\<T<sub>y</sub>\></sup> is distributed among the T<sub>x</sub> words in the text corpus.

 - When the length of the sequence T<sub>y</sub> increases, Beam search results in some numerical underflow. This is due to the fact that we maximize the product of all the conditional probabilities. One way of fixing this numerical issue is to maximize the log product of these probabilities (as the log is monotonically ascendent function), whitch is equivalent to maximizing the sum of the log probabilities. 
 
   In addition, because beam search is better doing with short sequences vs. longer sequences, we also need to normalize the maximization goal by (1/T<sub>y<sup>α</sup></sub>) where α ranges between 0 and 1:
   
     argmax<sub>Y</sub> (1/T<sub>y<sup>α</sup></sub>) x Sum<sub>t=1</sub><sup>\<T<sub>y</sub>\></sup> log(P(Y<sup>\<t\></sup> / X, {Y<sup>\<1\></sup>,..., Y<sup>\<T<sub>y</sub>-1\></sup>}=[{uplet<sub>1</sub>}, {uplet<sub>2</sub>},...,  {uplet<sub>3xT<sub>x</sub></sub>}]))
 
   Compared to exact serach algorithms such as BFS (Breadth Fist Search) and DFS (Depth First Search), Beam serach runs faster but does not guarantee exact maximum of P(Y/X)

### Attention model

## Conclusion

