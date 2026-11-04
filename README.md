# Reconstructing Message Recurrence in Anxiety-Related Social Media Discourse

### A Computational Social Science and NLP Research Project

**Research collaboration:** Northwestern University in Qatar  
**Research area:** Natural Language Processing · Computational Social Science · Information Circulation · Graph-Based Clustering

## Overview

How many social media posts represent genuinely distinct messages, and how many are variations of content that has already appeared elsewhere?

This project investigates textual recurrence in a large corpus of Facebook posts collected through anxiety-related search queries.

Rather than treating every post as an independent observation, the research develops and evaluates a pipeline for reconstructing *message families*: groups of posts that communicate substantially the same underlying message despite differences in wording, formatting, hashtags, or other surface features.

The corpus contains approximately 282,000 posts and over 250,000 normalized text records.

## Research Objectives

1. Identify recurring messages across large-scale social media discourse.
2. Distinguish substantive message reuse from topical similarity.
3. Reconstruct message families using lexical and semantic retrieval.
4. Identify and correct false merges caused by hashtags and recurring templates.
5. Examine observed recurrence and cross-page circulation.
6. Evaluate reconstruction quality through human annotation and robustness testing.

## Methodology

The research pipeline combines several complementary approaches.

### 1. Text Preprocessing

Posts are normalized to reduce superficial textual variation. Additional experiments investigate the effect of removing complete hashtag tokens and recurring boilerplate.

### 2. Lexical Candidate Retrieval

MinHash-based retrieval identifies potentially recurring messages using surface-form similarity.

Character n-gram similarity provides an additional lexical comparison.

### 3. Semantic Candidate Retrieval

E5 sentence embeddings represent posts in semantic space.

FAISS approximate nearest-neighbor search identifies additional candidate pairs that may express the same message despite differences in wording.

### 4. Graph-Based Message Reconstruction

Candidate pairs are evaluated using similarity thresholds and strong/weak edge rules.

Graph-based reconstruction groups posts into message families while restricting weak-only transitive chains that can cause unrelated messages to merge.

### 5. Cohesion Repair

Reconstructed families undergo cohesion diagnostics and repair to reduce incorrect groupings.

### 6. Human Validation and Robustness

The methodology is evaluated using:

- Stratified development audits
- Independent human annotation
- Held-out pairwise and triplet evaluations
- Cleaning and graph-construction ablations
- Candidate retrieval sensitivity analyses
- Holdout disjointness verification
- Alternative temporal specifications

## Selected Findings

A central finding is that surface-level features, particularly hashtags and recurring templates, can create misleading similarities between substantively different posts.

Removing these features can substantially improve message-family reconstruction quality.

In the held-out evaluation, reported pairwise F1 scores improved:

| Evaluation | Baseline | Cleaned |
|---|---:|---:|
| Annotator A | 0.619 | 0.776 |
| Annotator B | 0.653 | 0.860 |

These results concern the evaluated reconstruction configurations and should not be interpreted as universal performance estimates.

## Research Significance

Identifying recurring messages matters because treating repeated or slightly modified content as independent observations can distort measurements of online discourse.

This project explores a transparent methodology that may be applicable to other domains, including:

- Public health communication
- Misinformation research
- Political discourse
- Coordinated messaging
- Online information circulation

The reconstructed message families describe observed textual recurrence. They do not independently establish copying direction, causal diffusion, or clinical characteristics of the individuals involved.

## Repository Contents

`src/` contains the selected research implementations, including preprocessing, lexical and semantic candidate retrieval, graph construction, cohesion repair, validation, and downstream analysis.

`examples/` is reserved for synthetic demonstration data.

`results/` is reserved for approved aggregate results and visualizations.

## Reproducibility and Data Availability

This repository contains selected research code, not the underlying Facebook corpus.

The full experiments rely on source data and intermediate artifacts that are not distributed here. Some scripts require those artifacts and cannot be executed independently.

A public demonstration with synthetic data and full execution instructions may be added separately.

The code is provided for methodological inspection. Passing Python syntax checks does not establish end-to-end reproducibility.

## Research Status

Research code release in preparation. Further validation, documentation, and reproducibility checks are ongoing.

## Acknowledgment

Developed as part of a research collaboration with Northwestern University in Qatar.

## Contact

For questions concerning the implementation or research methodology, please open a GitHub issue.
