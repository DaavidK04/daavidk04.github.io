---
layout: post
title: "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding"
date: 2026-06-05
author: "David Kajkic & Tim Terbach"
---

## Motivation

Real-world data is, to a large extent, inherently graph-structured — ranging from knowledge graphs and recommendation systems to the web and e-commerce. These graphs are not just structural; their nodes and edges contain textual attributes, which defines them as textual graphs. While several approaches combining LLMs and GNNs already exist, they face two major limitations. First, converting a graph that consists of thousands of nodes and edges into a text sequence results in an excessive amount of tokens, which quickly surpasses the context window of most LLMs. Second, compressing an entire graph into one single embedding vector causes significant information loss, which leads to hallucination: the model makes up nodes and edges that do not even exist in the graph.

G-Retriever provides an effective approach to address these issues, relying on three core concepts:
- **GNN**: understands the graph structure
- **LLM**: understands the text and generates answers
- **RAG**: retrieves only the relevant subgraph instead of passing the entire graph to the LLM

The central concept is that answering a specific question requires only a small subset of the entire graph. G-Retriever uses the PCST algorithm to find this relevant subgraph, while keeping all relevant nodes and edges and still fitting into the LLM's context window.

Furthermore, the authors introduce GraphQA, a new benchmark test that evaluates question answering on real-world graphs.

## Table of Contents
- [Motivation](#motivation)
- [Table of Contents](#table-of-contents)
- [Background](#background)
- [GraphQA Benchmark](#graphqa-benchmark)
- [G-Retriever Architecture](#g-retriever-architecture)
- [Challenges: Hallucination \& Scalability](#challenges-hallucination--scalability)
- [Results](#results)
- [Critical Assessment](#critical-assessment)
- [Conclusion](#conclusion)
- [References](#references)

## Background

**GNN**:
A Graph Neural Network (GNN) is essentially a neural network tailored for graph structures. Instead of processing fixed inputs, it learns representations by having each node collect information from its neighbors. Through multiple layers of message passing, these local insights propagate across the graph, allowing the model to capture both local patterns and the global structure.

**RAG**:
Retrieval-Augmented Generation (RAG) is a method that improves LLMs by retrieving relevant information at inference time rather than depending just on the parametric knowledge of the model. Given a query, a retriever fetches the most relevant documents or data points from an external source, which are then passed to the LLM as further context for generating the response. This reduces hallucination and allows the model to process data that is outside of its context window or was not seen during training

**Soft Prompt Tuning**:
Fine-tuning an entire LLM is not practical in most cases – most modern models have billions of parameters, making full fine-tuning extremely expensive. Soft prompt tuning however, takes a different approach: it keeps the base model frozen and prepends a small set of trainable vectors to the input. Only these vectors are updated during the training while the rest of the model remains untouched. The result is a much cheaper fine tuning process while retaining the model's pre trained linguistic knowledge.



## GraphQA Benchmark

Since no suitable benchmark for graph question answering existed, the authors introduced GraphQA. Each entry in the benchmark follows a straightforward structure: a textual graph, a question, and an answer. The graph is provided in a CSV-like format listing all nodes and edges, and the model's job is to find the correct answer based on that.

GraphQA uses three datasets that increase in difficulty:

**ExplaGraphs**: Small graphs ($\approx 5$ nodes and $\approx 4$ edges), focused on commonsense reasoning. A typical question would be: *"Do argument 1 and argument 2 support or counter each other?"* The model has to understand basic relationships between concepts.

**SceneGraphs**: Medium-sized graphs ($\approx 19$ nodes and $\approx 68$ edges), describing objects and their spatial relationships within images. A typical question would be: *"Is there a woman to the right of the person behind the computer?"* 

**WebQSP**: Large scale knowledge graphs, requiring multi-hop reasoning across several edges. A typical question would be: *"What is the name of Justin Bieber's brother?"*

The three datasets are deliberately varied – the goal is to show that G-Retriever works across different graph types and sizes, not just one specific use case.

## G-Retriever Architecture

G-Retriever processes each query through a four-step pipeline.

1. **Indexing**: Before any query is processed, all nodes and edges are converted into embedding vectors and stored. This is done upfront, so the systen does not have to recompute the embeddings every time a new question comes in.
2. **Retrieval**: When a query arrives, it gets encoded the same way as the nodes and edges. (Top k most similar nodes, most relevant parts of the graph...)
3. **Subgraph construction**: Difference between RAG and G-Retriever, PCST (how deep should we explain that?)
4. **Generation**: The retrieved subgraph goes through two parallel paths. First, a graph attention network (gat) encodes the graph structure into a vector, which a small MLP (explain mlp and gat?) then maps into the LLM's vector space. Then the subgraph is converted into a text format listing nodes and edges, then concatenated with the query. Both are fed into the LLM which generates the final answer. 

## Challenges: Hallucination & Scalability

Two of the most critical limitations of existing approaches are hallucinations and poor scalability. G-Retriever tackles both through its RAG-based design.

**<u>Hallucinations</u>**:
- Baseline compresses entire graph into single embedding vector -> information loss-> LLM hallucinations nodes/edges
- G-Retriever retrieves actual nodes/edges directly from graph -> less hallucination
- Valid Nodes: 31% -> 77%, Valid Edges: 12% -> 76%, Fully Valid Graphs: 8% -> 62% (statistics relevant to the reader?)
- 38% still not fully valid -> hallucination reduced, not eliminated


**<u>Scalability</u>**:
- Converting full graph to text not feasible for large graphs
- WebQSP: $\approx 100000$ tokens before retrieval -> 610 after ($\downarrow$ 99%)
- Training time also reduces (again a lot of numbers, leave out or not?)
- Explain why pcst makes this possible? 
## Results

**<u>Performance</u>**:
- G-retriever beats all baselines on all three datasets
- G-Retriever + LoRA best configuration in total

**<u>Efficiency</u>**:
- SceneGraphs: 83% less tokens, 29% faster
- WebQSP: 99% less tokens, 67% faster

**<u>Hallucination Mitigation</u>**:
- 54% less hallucinations compared to baseline
- reference table 5?

**<u>Ablation study</u>**:
  
## Critical Assessment

- G-Retriever outperforms all baselines -> weak baseline -> no comparison against specialized knowledge graph systems that have been developed for years
- Hallucinations reduced BUT NOT ELIMINATED! – 38% still not fully valid
- Static procedure: pcst = fixed algorithm, not trainable. Authors admit this as limitation, suggest trainable retrieval 
- Only tested with Llama2-7B -> other llms?
- GraphQA benchmark uses existing datasets that were originally not designed for graph QA

## Conclusion

- G-Retriever is the first RAG-based approach for general textual graph QA
- Combines GNN, LLM and RAG effectively to tackle hallucination and scalability
- GraphQA benchmark fills a gap that previously existed in the field
- Results show strong performance across the graph types and sizes
- Static retrieval main limitation -> trainable retrieval logical next step
- Overall a solid foundation for future work on graph based question answering

## References
