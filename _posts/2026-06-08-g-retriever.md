---
layout: post
title: "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding"
date: 2026-06-05
author: "David Kajkic & Tim Terbach"
---

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>

## Motivation
"What is the name of Justin Bieber's brother?" Simple question – but the answer isn't written down in any single place. It's scattered across a web of connections: Justin, his parents, their children. To answer it, you have to follow the links. This is exactly the kind of problem G-Retriever is built for.
To understand how it works, we first have to understand how modern LLMs work. Every word is translated and mapped into a vector – a list of numbers that stores its meaning. The model is then trained on massive amounts of text so it learns how these tokens relate to each other. Language generation works sequentially: The model generates one token at a time, and it always picks the most likely next token.
This works remarkably well – as long as the input is bare text. However, not all data comes in a sequence of words. Real-world data is, to a large extent, stored as graphs. A graph is a structure that consists of nodes and edges, where nodes represent objects and edges the relationship. For example: "Justin Bieber" is connected to "Jaxon Bieber" through a "sibling" relation. 

![A small knowledge graph connecting Justin Bieber to his family through relations like sibling, parents, and children](/assets/images/bieber-graph.png)
*Adapted from Figure 2, [He et al. (2024)](https://arxiv.org/abs/2402.07630)*

Naturally, the next idea would be to just paste the graph and its values into the LLM. This is where two problems occur: First, large graphs would be converted into huge amounts of text, which would explode the LLMs context window. Second, compressing the graph into a single vector would result in information loss, which would cause the LLM to hallucinate nodes and edges. 

G-Retriever provides an effective approach to address these issues, relying on three core concepts:
- **GNN** (Graph Neural Network): understands the graph structure
- **LLM** (Large Language Model): understands the text and generates answers
- **RAG** (Retrieval-Augmented Generation): retrieves only the relevant subgraph instead of passing the entire graph to the LLM

The central concept is that answering a specific question requires only a small subset of the entire graph. G-Retriever uses the PCST algorithm to find this relevant subgraph, while keeping all relevant nodes and edges and still fitting into the LLM's context window.

Furthermore, the authors introduce GraphQA, a new benchmark test that evaluates question answering on real-world graphs.

## Table of Contents
- [Motivation](#motivation)
- [Table of Contents](#table-of-contents)
- [Background](#background)
- [G-Retriever Architecture](#g-retriever-architecture)
  - [Indexing](#indexing)
  - [Retrieval](#retrieval)
  - [Subgraph construction](#subgraph-construction)
  - [Answer Generation](#answer-generation)
    - [**Graph encoder (GAT)**](#graph-encoder-gat)
    - [**Projection (MLP)**](#projection-mlp)
    - [**Textualization**](#textualization)
    - [**Frozen LLM + Soft Prompt Tuning**](#frozen-llm--soft-prompt-tuning)
- [GraphQA Benchmark](#graphqa-benchmark)
- [Results](#results)
- [Critical Assessment](#critical-assessment)
  - [Strengths](#strengths)
  - [Limitations](#limitations)
- [Conclusion](#conclusion)
- [References](#references)

## Background

**GNN**:
A Graph Neural Network is essentially a neural network tailored for graph structures. Instead of processing fixed inputs, it learns representations by having each node collect information from its neighbors. Through multiple layers of message passing, this information propagates across the graph, allowing the model to capture both local patterns and the global structure. 
However, a GNN only delivers vectors, not a textual answer. This is where LLMs come in handy.

**LLM**:
As already mentioned in the motivation section, a Large Language Model generates one token after another. This mechanism is called autoregressive generation. It always chooses the most likely next token given everything written so far. In short, we define this as:

$$p(y_i \mid y_1, y_2, \dots, y_{i-1})$$

where $y_i$ is the next token and $y_1 ... y_{i-1}$ are all previous tokens. So the probability of the next token is dependent on all earlier tokens.
Although LLMs are strong with language, they only work well as long as the input fits into the context window – a whole graph does not – which is the problem RAG is built to solve.

**RAG**:
Retrieval-Augmented Generation is a method that improves LLMs by retrieving relevant information at inference time rather than depending just on the parametric knowledge of the model. Given a query, a retriever fetches the most relevant documents or data points from an external source, which are then passed to the LLM as further context for generating the response. Without RAG, the LLM would try to reconstruct everything from its parametric knowledge, which causes hallucination. With RAG, it reads the retrieved information directly, instead of reconstructing it from memory. This is what reduces hallucination in the first place.

The RAG described so far is just the generic version for text and documents. The question arises how it can be applied to graphs, since the graph structure now has to be taken into consideration. This is exactly the problem G-Retriever solves.

## G-Retriever Architecture

Now, all three components from the background come together. G-Retriever processes each query through a four-step pipeline.

![The G-Retriever pipeline: indexing, retrieval, and subgraph construction feed into answer generation, where a graph-encoded vector and the textualized subgraph are combined in a frozen LLM](/assets/images/pipeline.png)

### Indexing
Before any query is processed, all nodes and edges are converted into embedding vectors and stored:

$$z_n = \text{LM}(x_n)$$

Here, $x_n$ is the text at a certain node $n$, $LM$ is the embedding model (SentenceBERT in this case), and $z_n$ is the resulting embedding vector.
This is done upfront, so the system does not have to recompute the embeddings every time a new question comes in. The computed embeddings are then stored into a Nearest-Neighbor-Structure. With that, the whole graph is represented as searchable vectors. But the actual search – the crucial part that connects the user's question to the graph – is still missing. The next step deals with filling the gap from a query to the relevant nodes.


### Retrieval
The query is vectorized by the same embedding model, ensuring that the query and all nodes end up in the same vector space and can be compared.

$$z_q = \text{LM}(x_q)$$

As before, $LM$ is the embedding model and $z_q$ is the embedding vector. The only difference is that $x_q$ is the query, and not one specific node. 
Since query and nodes are now comparable, their similarity can be measured through cosine similarity:

$$\cos(z_q, z_n) = \frac{z_q \cdot z_n}{\|z_q\| \, \|z_n\|}$$

To calculate the similarity between a node and the query, their scalar product is divided by the product of both vectors' lengths. By dividing out the lengths, only the direction of the vectors matters.  The result is a value between -1 and 1. A value of -1 means the vectors point in opposite directions, so they are unrelated, while a value near 1 means the node and the query are similar. However, since a graph contains a large number of nodes and edged, only the most relevant ones are filtered out:

$$V_k = \operatorname*{arg\,topk}_{n \in V} \cos(z_q, z_n)$$

$$E_k = \operatorname*{arg\,topk}_{e \in E} \cos(z_q, z_e)$$

This may look confusing at first, but both formulas do essentially the same thing: $\operatorname*{arg\,topk}$ takes the $k$ elements with the highest values out of all nodes $n$ in the node set $V$ (same goes for all edges $e$ in $E$). So all $k$ nodes and edges with the highest similarity to the query are kept.

The retrieval passes loose, scattered hits. The naive idea is to just take the top k nodes and edges in between. However, this is not enough since the hits are spread out across the graph and are not related to each other, and a handful of loose items does not result in a coherent subgraph that can be passed on as context. The next step is to turn these scattered hits into a compact and connected subgraph.

### Subgraph construction

The solution is called PCST – Prize-Collecting Steiner Tree. The name consists of two parts:

- Steiner tree: Connects a set of relevant nodes into a connected tree. Intermediate nodes are allowed to be included to connect the relevant nodes, even though they are not relevant themselves.
- Prize-Collecting: Not every node has to be connected necessarily. Every node has a prize (reward), every edge has a cost (price). The algorithm then decides which nodes are worth including, weighing their prizes against the edge cost needed to connect them.

PCST balances two opposing goals: maximizing the collected relevance while staying small and coherent. 

In G-Retriever, the prize is set to the relevance. The top k nodes and edges with a high cosine similarity receive prizes according to their similarity, the higher the similarity, the higher the prize. The rest outside of the top k gets 0. The cost on the other hand reflects the size: Every edge has its own cost ($C_e$), which stops the subgraph from growing uncontrollably. 

Formally, this can be written as

$$S^* = \arg\max_{S \subseteq G,\; S \text{ connected}} \left( \sum_{n \in V_S} \text{prize}(n) + \sum_{e \in E_S} \text{prize}(e) - \text{cost}(S) \right)$$

The arg max searches for the subgraph S that maximizes the expression in the brackets, just like the arg top k from the retrieval step. The condition $S \subseteq G, S\ connected$ restricts the search to connected subgraphs of $G$, which is what guarantees that the result is one connected subgraph instead of scattered pieces. The two sums add up the prizes of all included nodes and edges. From this, the cost is subtracted, so the final score rewards relevance and penalizes size.

The cost is defined as:

$$\text{cost}(S) = |E_S| \times C_e$$

The number of edges in the subgraph is multiplied by the cost per edge $C_e$. More edges mean higher cost and a larger subgraph, which is penalized. This is what keeps the subgraph compact.



### Answer Generation

The PCST outputs a compact subgraph, however, a subgraph is still not an answer. It is passed to the LLM through two parallel ways – as a structure (vector) and as text. Both are necessary, because they carry different information: The vector carries the overall structure, the text carries the concrete attributes – actual names, values, and relations as readable text. An LLM with just the vector would know the structure but not the details; with just the text it would lose the structural overview. 

#### **Graph encoder (GAT)**

The subgraph goes through a Graph Attention Network. A GAT is a neural network for graphs: it allows nodes to weigh how much they care about their neighbors, encoding the structural information into vector representations. As a result, the whole subgraph is pooled into a single vector. 

Formally, this can be described as

$$h_g = \text{POOL}(\text{GNN}_{\phi_1}(S^*))$$

where $S^*$ is the subgraph from PCST, $\text{GNN}_{\phi_1}$ is the graph encoder and $\phi_1$ are the trainable parameters. $\text{POOL}$ is the operation that summarizes all node-vectors to a single vector, because the LLM later gets one graph-vector as its prompt, not hundreds of vectors. 
In most cases, it is the mean value of all nodes. $h_g$ is the result: the single vector that represents the whole subgraph. Still, $h_g$ is not something the LLM can read yet. So we need something to translate it into the vector space of the LLM.

#### **Projection (MLP)**

A small neural network, called multi-layer perceptron (MLP), translates $h_g$ into the vector space of the LLM, since the graph encoder and the LLM work in different vector spaces. The result is a "graph-token" which is attached to the LLM-Input.

$$\hat{h}_g = \text{MLP}_{\phi_2}(h_g)$$

Here, $h_g$ is the graph vector from earlier, $\text{MLP}_{\phi_2}$  is the projection-MLP, and $\phi_2$ are its trainable parameters. It is essentially the counterpart to $\phi_1$. $\hat{h}_g$ is the result: the same graph vector, now in the vector space of the LLM. This graph-token acts as a soft prompt: a small, trainable input that steers the LLM without changing its weights.

#### **Textualization** 

In parallel, the same subgraph is textualized into a CSV-like list of nodes and edges.

Nodes:

| node_id | node_attr     |
|---------|---------------|
| 15      | justin bieber |
| 294     | jaxon bieber  |
| 356     | jeremy bieber |


Edges:

| src | edge_attr              | dst |
|-----|------------------------|-----|
| 15  | people.person.parents  | 356 |
| 356 | people.person.children | 294 |

This text is then combined with the question and passed to the LLM.

#### **Frozen LLM + Soft Prompt Tuning**

Lastly, both sides from the pipeline are merged and put into the LLM. The LLM then generates the answer, token by token, via its autoregressive loop. 

$$p_{\theta, \phi_1, \phi_2}(Y \mid S^*, x_q) = \prod_{i=1}^{r} p_{\theta, \phi_1, \phi_2}(y_i \mid y_{<i}, [\hat{h}_g; h_t])$$

Here, $$p_{\theta, \phi_1, \phi_2}(Y \mid S^*, x_q)$$ is just the probability of the answer $$Y$$, given the subgraph $$S^*$$ and the query $$x_q$$. The three indices are all involved parameters – $$\theta$$ the LLM, $$\phi_1$$ the GAT, $$\phi_2$$ the MLP. Only the parameters of the GAT and the MLP are trained.

$\prod_{i=1}^{r}$ is the product over all tokens plus $p(y_i \mid y_{<i})$ – this is the same autoregressive loop from the background, now conditioned on the graph.

$[\hat{h}_g; h_t]$ is the concatenation of both approaches from the pipeline before, $\hat{h}_g$ is the graph-token and $h_t$ is the textualized graph plus the question.

Because only $\phi_1$ and $\phi_2$ are trained, this training method is called soft prompt tuning: the soft prompt (and the graph encoder) are trained, while the LLM itself stays untouched. This ensures efficiency while maintaining the LLM's language ability.

Let's walk through the whole pipeline with a familiar example.

**1. Indexing**

Every node and edge is converted into embeddings. "justin bieber", "jaxon bieber", "jeremy bieber", "parents", "children" are all vectorized. The intermediate result is a list of all nodes and edges, represented as vectors.

**2. Retrieval**

The query "What is the name of Justin Bieber's brother?" is also vectorized. Its cosine similarity is computed against all nodes and edges. The most similar nodes and edges will be "Justin Bieber" and the relation edges like 'parents' and 'children', since "Justin Bieber" literally appears in the query, and 'parents' and 'children' are semantically close to 'brother'. These top hits are kept – but they are just loose pieces, not a connected graph yet. However, this intermediate retrieval result is the exact setup for PCST.

**3. PCST**

Prizes are assigned: The relevant hits from retrieval get high prizes. 

The first key trick – the connector node: "Jeremy Bieber" is not directly relevant on its own, since he is not in the query, but he is needed to connect Justin to the rest of the graph. 

The second key trick – bringing along the answer node via structure: "Jaxon Bieber" is the actual answer, even though he does not appear in the query at all. He makes it into the subgraph because he is connected to the relevant part via "children".

Intermediate result: A small, connected subgraph: Justin $\rightarrow$ Jeremy $\rightarrow$ Jaxon


**4. Answer generation**

The subgraph goes through both pipeline paths: as a vector (GAT $\rightarrow$ MLP $\rightarrow$ graph-token) and as text (the CSV table from earlier). Both, along with the question, go into the frozen LLM. The LLM produces the correct answer: Jaxon Bieber.

The output is now set, but how good is it? Evaluating this requires a proper benchmark.

## GraphQA Benchmark

Since no suitable benchmark for graph question answering existed, the authors introduced GraphQA. Each entry in the benchmark follows a straightforward structure: a textual graph, a question, and an answer. The graph is provided in the CSV-like format we saw earlier, and the model's job is to find the correct answer based on that.

GraphQA uses three datasets that increase in difficulty:

![Three example entries from the GraphQA benchmark: an explanation graph, a scene graph, and a knowledge graph, each with its graph structure, textualized form, question, and answer](/assets/images/graphqa-examples.png)
*Figure 2 from [He et al. (2024)](https://arxiv.org/abs/2402.07630)*

**ExplaGraphs**: Small graphs ($\approx 5$ nodes and $\approx 4$ edges), focused on commonsense reasoning. A typical question would be: *"Do argument 1 and argument 2 support or counter each other?"* The model has to understand basic relationships between concepts. The measured metric is **accuracy**.

**SceneGraphs**: Medium-sized graphs ($\approx 19$ nodes and $\approx 68$ edges), describing objects and their spatial relationships within images. A typical question would be: *"Is there a woman to the right of the person behind the computer?"*  The measured metric is **accuracy**.

**WebQSP**: Large scale knowledge graphs ($\approx 1400$ nodes), requiring multi-hop reasoning across several edges. A typical question would be: *"What is the capital of the country Ronaldo was born in?"* The measured metric is **Hit@1**: it checks whether the highest scoring result matches the ground truth, which matters because a question can have several valid answers.

The three datasets are deliberately varied – the goal is to show that G-Retriever works across different graph types and sizes, not just one specific use case.
Now, we have the benchmark and the metrics – let's take a look at how G-Retriever actually performs.

## Results

G-Retriever was tested in three different configurations.

- Inference-only: It merely relies on a fully frozen LLM to answer the questions directly, without any training.
- Frozen LLM + Prompt Tuning: The LLM stays frozen, only the prompts get trained. This is the soft prompt tuning from the architecture section.
- Tuned LLM (LoRA, an efficient fine-tuning method): The LLM itself gets trained.

The results show a consistent picture across all three configurations. 

**Performance**

G-Retriever beats all baselines on all three datasets. The gains are largest in the frozen LLM with prompt tuning – accuracy improves by roughly 28% to 47% over the baselines. The best overall results come from combining G-Retriever with a tuned LLM. The most striking part is that even the frozen LLM variant performs strongly: the model still beats approaches that were trained, even though the LLM was not touched at all. This is exactly where soft prompt tuning pays off.


**Efficiency**

Because G-Retriever only keeps the relevant subgraph, the data the LLM has to process drops dramatically:

| Dataset     | Token reduction | Node reduction | Training time |
|-------------|-----------------|----------------|---------------|
| WebQSP      | −99%            | −99%           | −67%          |
| SceneGraphs | −83%            | —              | −29%          |

The most striking part is the payoff that PCST provides. For WebQSP, the token usage was reduced from $\approx$100,000 tokens to $\approx$600 tokens. This is more than just a numerical reduction – as it makes working with large graphs possible in the first place. A WebQSP graph would not fit in the context window – our problem from the motivation section. The reduction stands out less for smaller graphs: For SceneGraphs, the token usage was reduced by "only" 83%, which shows that PCST provides more benefits for larger graphs.

**Hallucination**

The number of fully valid graphs (answers where all referenced nodes and edges actually exist in the graph) increased from 8% for the baseline to 62% for G-Retriever. This confirms the statement from the background: RAG reduces hallucinations, because the model can read the facts directly, instead of relying on its parametric knowledge. However, 62% means that 38% are not fully valid. Hallucinations are reduced dramatically, but not eliminated.

**Ablation study**
  
An ablation study evaluates the impact of components by removing them and measuring the resulting change in performance. 

| Removed component | Performance drop |
|-------------------|------------------|
| Graph encoder     | −22.5%           |
| Textualized graph | −19.2%           |
| Retrieval         | −9.4%            |

The table shows that all three components matter – especially the graph encoder and the textualized graph. Leaving either one out hurts, which confirms our point in the architecture: both matter. Retrieval also contributes, though less dramatically. This makes sense, because without retrieval, the model still receives both information sources, just with more irrelevant nodes and edges around them. Removing the textualized graph or the graph encoder, on the other hand, takes away an entire source of information.

These results are undoubtedly impressive, but a closer look reveals that performance numbers alone rarely catch the whole picture. Examining the remaining limitations is essential for a complete assessment.


## Critical Assessment

Despite its strong results, like every other method, G-Retriever is not without limitations. Understanding both its strengths and weaknesses provides a more balanced view of the approach.

### Strengths

As earlier mentioned, G-Retriever drastically reduces the number of input tokens by up to 99%, which makes it possible to process graphs that would otherwise exceed the context window. By retrieving facts directly from the knowledge graph, G-Retriever grounds its responses in structured data rather than relying on its parametric knowledge. As a result, the number of fully valid answers increases from 8% to 62%. The ablation study shows that both the graph encoder and the textualized graph contribute significantly to the performance. This confirms that the pipeline structure is genuinely justified by evidence, not just a nice-sounding design choice.

### Limitations

While validity is improved significantly, hallucinations remain a challenge. The evaluation shows that 38% of all answers are not fully valid, which means that nodes or edges were hallucinated. For real world applications where factual correctness is critical, this unreliability limits G-Retriever's readiness.

The current retrieval process relies on a fixed PCST algorithm rather than a trainable retrieval module. PCST cannot adapt its retrieval strategy based on feedback. The authors themselves identify this lack of trainability as an important limitation and a potential area for future research.

The retrieval also heavily relies on the similarity between the user query and the graph's attributes. Mismatches in phrasing can cause relevant information to never get retrieved in the first place, and nothing in the pipeline is trained to fix that mismatch. It's a classic "garbage in, garbage out" risk.

Lastly, the PCST algorithm is NP-hard in general. In practice, G-Retriever uses an efficient approximation, so it runs fine on the tested datasets. But for very large or constantly changing graphs, this step could still become a bottleneck.

Overall, these limitations show that G-Retriever is not a complete solution to graph reasoning yet. But it is an important step in that direction. By demonstrating how graph structure and language models can synergize, the paper provides a strong foundation for further development.

## Conclusion

G-Retriever is the first RAG-based approach for general textual graphs. By combining graph neural networks, large language models, and retrieval-augmented generation, it introduces a framework for question-answering over text-attributed graphs. The results show that this approach can outperform existing baselines, scale to larger graphs, and improve answer grounding. At the same time, limitations such as remaining hallucinations and static retrieval highlight important leverage points for future research. Yet, G-Retriever provides a strong foundation for interacting with graphs as naturally as we interact with LLMs – bringing the idea of "chatting with your graph" closer to reality.

## References

[1] He, X., Bresson, X., Laurent, T., Perold, A., LeCun, Y., & Hooi, B. (2024). *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering.* arXiv:2402.07630. [https://arxiv.org/abs/2402.07630](https://arxiv.org/abs/2402.07630)

[2] Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

[3] Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* arXiv:1908.10084. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

[4] Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* arXiv:2307.09288. [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)

[5] Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
