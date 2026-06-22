# Named Entity Recognition and Classification - NERC Lab

**Rule-Based Information Extraction on Wikipedia Articles**

![figure.jpg](figure.jpg)

## About

This repository presents a practical implementation of a Named Entity Recognition and Classification system.

The objective is to extract meaningful named entities from Wikipedia articles and classify them into predefined semantic categories.

Rather than using a pretrained black-box model, this project focuses on a rule-based approach in order to understand the fundamental logic behind entity extraction:

* How raw text can be transformed into structured information
* How named entities can be detected using textual patterns
* How contextual clues help classify entities
* How precision and recall reflect different extraction behaviors
* How rule-based NLP systems can be improved through error analysis

The lab is conducted on a Wikipedia corpus and evaluated against a gold standard annotation file.

## Task Definition

Named Entity Recognition and Classification consists of two steps:

1. **Entity recognition**: detecting relevant spans of text.
2. **Entity classification**: assigning a semantic category to each detected entity.

Example:

```text
"Barack Obama was born in Honolulu in 1961."
```

Expected extraction:

```text
Barack Obama -> PER
Honolulu     -> LOC
1961         -> TMP
```

The objective is to produce structured outputs from unstructured text.

## Entity Classes

The system predicts five entity types:

```text
PER  -> Person
LOC  -> Location
ORG  -> Organization
TMP  -> Time / Date / Period
MISC -> Miscellaneous named entity
```

These categories cover the main types of information commonly extracted in classical NLP pipelines.

## Data

The lab relies on two main input files:

```text
wikipedia-corpus.txt
student-gold-standard.tsv
```

The first file contains Wikipedia articles to process.
The second file contains reference annotations used for evaluation.

The program generates:

```text
results.tsv
```

This output file contains the detected entities and their predicted labels.

## Output Format

Each prediction follows the format:

```text
article_title    extracted_entity    predicted_class
```

Example:

```text
Paris_Article    Paris        LOC
Obama_Article    Barack Obama PER
History_Article  1945         TMP
```

## Methodological Approach

This project uses a rule-based NLP approach.

Instead of training a statistical or neural model, the system relies on handcrafted rules based on:

* Capitalization patterns
* Regular expressions
* Date and year formats
* Entity suffixes
* Contextual keywords
* Known names, locations and organizations
* Wikipedia-style textual structures

The goal is not only to obtain predictions, but to understand why an entity is detected and why a label is assigned.

## Extraction Pipeline

The global pipeline is:

```text
1. Read Wikipedia articles
2. Detect candidate named entities
3. Filter noisy candidates
4. Classify each entity
5. Write predictions to results.tsv
6. Evaluate predictions against the gold standard
```

This pipeline illustrates a classical information extraction workflow.

## Rule-Based Entity Detection

Candidate entities are mainly detected through textual patterns.

Examples of useful signals:

```text
Capitalized words          -> possible named entities
Multi-word expressions     -> possible person, organization or location
Years and date expressions -> TMP
Acronyms                   -> possible ORG
Wikipedia titles/context   -> useful disambiguation clues
```

For example:

```text
"University of Oxford" -> likely ORG
"New York"             -> likely LOC
"August 2006"          -> likely TMP
```

## Entity Classification

After extracting candidates, the system assigns a class.

Typical rules include:

```text
University, Company, Association -> ORG
City, River, County, Province    -> LOC
Mr., Dr., first names            -> PER
Year, month, century             -> TMP
Nationality, language, event     -> MISC
```

The classification step combines direct lexical clues with contextual interpretation.

## Mathematical Perspective

A text can be represented as a sequence of tokens:

```text
x = (w1, w2, ..., wT)
```

The goal of NERC is to extract a set of entities:

```text
E = {(entity_1, label_1), ..., (entity_n, label_n)}
```

where each label belongs to:

```text
Y = {PER, LOC, ORG, TMP, MISC}
```

Unlike token-level BIO tagging, this lab directly outputs detected entity spans with their final class.

## Evaluation Metrics

The system is evaluated using classical information extraction metrics.

### Precision

Precision measures how many predicted entities are correct:

```text
Precision = correct predictions / total predictions
```

High precision means the system avoids false positives.

### Recall

Recall measures how many expected entities were found:

```text
Recall = correct predictions / expected entities
```

High recall means the system retrieves many of the gold-standard entities.

### F-0.5 Score

The F-0.5 score combines precision and recall while giving more importance to precision.

This is useful when false positives are considered more harmful than missing some entities.

## Key Observations

This lab highlights several important points:

* Rule-based systems can be strong baselines for structured NLP tasks
* Entity extraction depends heavily on surface patterns and context
* Precision and recall are often in tension
* Adding too many rules may improve local performance but reduce generalization
* Error analysis is essential for improving NLP systems
* NERC is a bridge between raw text and structured knowledge

## Core Takeaway

Named Entity Recognition and Classification is a fundamental NLP task.

It transforms raw text such as:

```text
"Apple was founded by Steve Jobs in California in 1976."
```

into structured information:

```text
Apple       -> ORG
Steve Jobs  -> PER
California  -> LOC
1976        -> TMP
```

This project shows how simple linguistic rules, regular expressions and contextual patterns can already extract meaningful knowledge from text.

## Dependencies

```text
Python 3.x
re
csv
```

No large pretrained model is required.

---
***Alexandre Mathias DONNAT, Sr***
