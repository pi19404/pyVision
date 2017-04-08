---
layout: post
title: Apache OpenNLP Installation on ubuntu 14.04
category: Software Installation
---

**Apache OpenNLP Installation on ubuntu 14.04**
------------------------------------


The Apache OpenNLP library is a machine learning based toolkit for the processing of natural language text.

It supports the most common NLP tasks, such as tokenization, sentence segmentation, part-of-speech tagging, named entity extraction, chunking, parsing, and coreference resolution. These tasks are usually required to build more advanced text processing services. OpenNLP also includes maximum entropy and perceptron based machine learning.


**Download**

you can download opennlp source from

```
git clone https://github.com/apache/opennlp

```


**Pre Requisites**

openlp requires `Apache maven` build manager for Java projects.

```
sudo apt-get install maven maven2
```

**compile and install**

execute the following command to compile `opennlp`

```
mvn compile
mvn install
```

**IDE**

`opennlp` projects can be accessed via IDE's like netbeans or eclipse

