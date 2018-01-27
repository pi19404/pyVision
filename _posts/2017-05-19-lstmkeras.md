---
layout: post
title: NLP Introduction to LSTM using Keras
published: true
---


# **Long Short-Term Memory Network**

The Long Short-Term Memory network, or LSTM network, is a recurrent neural network.

RNN are networks with loops in them, allowing information to persist.Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to lear

We can use LSTM for host of NLP classification problem.

## **Input Training data**

The input training data is of the form
```
{ u'text': u'11 dollar to rupee conversion', u'intent': u'"conversion"'}
```

## **Word Embedding**

The input to LSTM network is a sequence of tokens of the sentense and the output is associated class lable.
The LSTM network will model how various words belonging to a class occur in a statement/document.

We will be using spaCy NLP package.The built in word embedding function  provides a word vector of length 300.

For details about using spaCy and computing word vectors refer to the article http://34.194.184.172/emotix/index.php/2017/05/13/nlp-spacy-word-and-document-vectors/

```
    def word2vec(self,tokens):
        output=[]
        for d in tokens:
            doc = self.nlp1(d['text'])
            t=[]
						l=[]
            for token in doc:
				  t.append(token.vector)								
			      l.append(token.lemma)
			d['tokens']=l
            d['features']=t;
            output.append(d)
        return output
```

Below is the code on how to call the above function

```
vectors=nlp.word2vec(data)
print("number of training samples are", len(vectors))
print("dimension each token of training data ", len(vectors[0]['features'][0]))
```

The above function accepts the training data.For each sentence in the training data the function computes the document vector for each token and stores the list of features in the `features` tag of the output `json` data.

## **Pre-Processing  the sequential data**

Since we will be analyzing short sentences and not long paragraphs of text we will choose the maximum number of tokens in the sentence to be 40

If the tokens in a sentence are less than 40 we will be zero padding or trimming it so that all sequence are of the same length.This step pre-processes the sequential data so that training data consists of vectors of the same dimension.

```
    def pad_vec_sequences(self,sequences, maxlen=40):
        new_sequences = []
        for sequence in sequences:
            orig_len, vec_len = np.shape(sequence)

            if orig_len < maxlen:
                new = np.zeros((maxlen, vec_len))
                new[maxlen - orig_len:, :] = sequence
            else:
                new = sequence[orig_len - maxlen:, :]
            new_sequences.append(new)

        return np.array(new_sequences)
```				

The above function is used as follows

```
maxlen=40
Xtrain=[]
labels=[]
for d in vectors:
   Xtrain.append(d['features'])
   labels.append(d['intent'])
Xtrain = self.pad_vec_sequences(Xtrain, maxlen)						
```						

The above function pads or truncates the input training data so that each sample of training data is of size `(maxlen,dimension)`
Thus inputs to the neural network are of fixed dimension vector of size (40,300) representing the sentence/document being analyzed.


We will prepare the training data in a form suitable to be used with keras

First we will process the output labels so that textual categories are covnverted to one hot encoded vector form

```
from sklearn import preprocessing
from keras.utils import np_utils, generic_utils
				
self.label_encoder = preprocessing.LabelEncoder()
#converts text categories to integers
self.label_encoder.fit(labels)
y=self.label_encoder.transform(labels)
        
#converts integers to one hot encoded vectors
y=np_utils.to_categorical(y)
```				

Our training data consists of 3 unique classes so the output vectors are encoded as follows

```

[1  0  0]
[0  1  0]
[0  0  1]

```

## **Training and Cross Validation Data Split**


No we will split our training data into training and cross validation data

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, random_state = 42)
```

This splits the training data such that 70% data is used for cross validation and 30% data is training data.

If the input training data X has 282 samples then `X_train` will have 84 and `X_test` will have 198 samples.

**Keras Pre Processing**

The training data required for keras is of the form `[samples, time steps, features]`

In our case time steps is length of sequential data being analyed ie maximum number of words being analyzed in the sequence.

```
 X_train = numpy.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2]))
```

**Keras Configuration**

Keras is a high-level neural networks API, written in Python and capable of running on top of either TensorFlow or Theano

We will be using tensorflow as backend to Keras

The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers

```
model = Sequential()
```

Stacking layers is as easy as `.add():`

In the present example we want to configure Keras to use LSTM networks.Hence first layer we want to stack is LSTM

```
 model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
```

The LSTM network will accepts input vectors of shape `(40,300)`


The output of the LSTM network is class lable.Hence number of outputs of the last layer is equal to the number of unique classes.In the present example the number of output classes are 3.

```
model.add(Dense(nclasses, activation='softmax'))
```

We are specyfying that we will be using softmax activation function for the output layers

And then we can choose the how we want to model the hidden layers between the input and output layers

```

        model = Sequential()
        model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(20))
        model.add(Dense(nclasses, activation='softmax'))
```

Once the layers of model are added we configure the parameters for the learning process

```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

We choose `categorical_crossentropy` as loss function when we want to perform categorical classification task.We choose the optimizer as `adam optimizer` which is a slightly enhanced version of the stochastic gradient descent.For any categorical classification problem the `metrics` parameter is set to `accuracy`

**Training the model**

To train the model 

```
 model.fit(X_train,y_train,epochs=10, batch_size=1, verbose=2)
```

we can choose the number of epochs and batch size so that training data can be iterated over in batches.

**Evaluating the model**

```
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print loss_and_metrics
model.save('my_model.h5')
```

**Batch Size**

The Keras implementation of LSTMs resets the state of the network after each batch.

This suggests that if we had a batch size large enough to hold all input patterns and if all the input patterns were ordered sequentially, that the LSTM could use the context of the sequence within the batch to better learn the sequence.

Keras shuffles the training dataset before each training epoch. To ensure the training data patterns remain sequential, we can disable this shuffling.

**References**

- https://keras.io/getting-started/sequential-model-guide/
- https://keras.io/
