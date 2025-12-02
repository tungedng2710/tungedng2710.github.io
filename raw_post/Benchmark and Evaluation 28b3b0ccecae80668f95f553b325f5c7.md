# Benchmark and Evaluation

**Table of contents**

## **1. OmniDocBench**

### **1.1. Dataset: Types, scale, and diversity**

- OmniDocBench is a curated evaluation benchmark composed of nine distinct document types (e.g. academic papers, textbooks, exam papers, newspapers, financial reports, slides, handwritten notes)
- It contains about1,355 PDF pages in the version hosted on GitHub (v1.5) covering 9 varieties of documents, 4 layout types, and 3 language types (English, Chinese, and maybe mixed)
- The annotations are rich and fine-grained: • Block/region–level: over 20,000 block annotations (boxes of text, images, tables, formula regions, etc.)  • Span/inline‐level: more than 70,000 span-level annotations (inline formulas, subscript, text lines, etc.)  • For tables: both LaTeX and HTML forms are provided (so models can be evaluated under different representations)
- Each page also carries page-level attributes (5 tags), text-block attributes (3), and table-block attributes (6 “special issue” binary flags) to allow attribute-based evaluation analyses
- Reading order is annotated, i.e. how blocks should be ordered/sequenced when linearizing the page.

### **1.2. Annotation process & quality control**

- They start with automatic/“intelligent pre-annotation” (using existing layout and OCR tools) → then human annotators correct bounding boxes, content, order, etc. → followed by expert review / quality checks.
- Because of variations (e.g. scanned/fuzzy places, watermarks) they also include “special cases” in their attribute tags to force methods to deal with real-world nuisances.

**Benchmark / evaluation metrics & methodology**

**The methodology has three levels:**

1. End-to-end evaluation — treat the model as a black box from PDF → structured output (e.g. Markdown/HTML or a structured format).
2. Module / component-level evaluation — isolate tasks like layout detection, OCR/text extraction, table parsing, formula recognition, reading-order.
3. Attribute / per-document / per-block analysis — break results down by document types, by block attributes (e.g. “tables with merged cells,” “watermark present,” languages) to see where methods fail.
    - Matching / alignment: when predicted blocks or spans differ in how they segment (split/merge) from ground truth, they use a hybrid matching algorithm that allows alignment by normalized edit distance or adjacency search. This helps avoid punishing minor segmentation differences too harshly.

**Key parts of matching and metrics:**

- Text / OCR metrics: normalized edit distance (lower is better) for text at block or span level. Also BLEU/METEOR are also supported in the evaluation tool.
- Table recognition metrics: they use TEDS (Tree-Edit-Distance-based Similarity) to compare table structure + content. There is also a “TEDS-S” (simplified) measure in some cases.
- Formula recognition: they propose a metric named CDM (Complex Document Matching) to evaluate formulas, intended to better handle variation in LaTeX or equivalent representations when surface text differs. (Higher is better)
- Reading order: also evaluated by edit distance between predicted reading order and ground truth order of blocks. Lower is better.
- Overall metric: In the GitHub version, they compute an “Overall” score by averaging three normalized sub-scores: (1 − normalized text edit distance) × 100, table TEDS, formula CDM. Thus each subtask contributes equally to the overall.

### **1.3. Results**

| **Method** | **Overall** | **Text (ED)** | **Formula** | **Table** | **RO (Edit)** |
| --- | --- | --- | --- | --- | --- |
| MinerU2.5 (1.2B) | 90.67 | 0.047 | 88.46 | 88.22 | 0.044 |
| MonkeyOCR-pro-3B | 88.85 | 0.075 | 87.25 | 86.78 | 0.128 |
| dots.ocr (3B) | 88.41 | 0.048 | 83.22 | 86.78 | 0.053 |
| MonkeyOCR-3B | 87.13 | 0.075 | 87.45 | 81.39 | 0.129 |
| MonkeyOCR-pro-1.2B | 86.96 | 0.084 | 85.02 | 84.24 | 0.130 |

## **2. PubTabNet**

The authors build a large-scale dataset to support image-based table recognition (i.e. given just an image of a table, recover its structure + cell text) rather than relying on PDF source code or markup.

### **2.1. Dataset**

**Overview**

- PubTabNet comprises about568,000 table images automatically extracted from scientific articles in the PubMed Central Open Access (PMCOA) subset.
- Each table is paired with a ground-truth HTML representation that encodes its structural markup (rows, columns, headers) plus the content of non-empty cells.
- The process: align the XML (or structured) version of the article with the PDF to find the table region and its markup, then render (or crop) the table image and link it to the HTML structure.
- The dataset also classifies cells into header vs body cells, which helps downstream usage.

**Comparisons & qualities**

- Compared to prior datasets (SciTSR, TableBank, etc.), PubTabNet’s strengths are:
    - Diversity — tables from thousands of journals → many layout/format styles.
    - Combined structure + content — not just layout, but the actual cell text (for non-empty cells).
    - HTML representation — output format is HTML, which is a tree structure (tags, nested rows/columns) and thus more expressive.

### 2.2. Metrics: **TEDS (Tree-Edit-Distance-based Similarity)**

**Representing a table as a tree**

```html
<table>
  <tr><td>Apple</td><td>Banana</td></tr>
  <tr><td>Cherry</td><td>Date</td></tr>
</table>
```

This is a **tree**:

- The root node is <table>.
- It has two child <tr> nodes (rows).
- Each <tr> has two <td> children (cells).
- Each <td> has a text node as its leaf (the cell content).

**Edit operations and their costs**

TEDS applies the Tree Edit Distance (TED) algorithm, a generalization of the string edit distance (Levenshtein).

Allowed operations are:

1. **Insert** a node
2. **Delete** a node
3. **Replace** a node (e.g., changing a <td>’s text content)

Each operation has a cost. The total cost is the minimum number of edits to turn the predicted tree into the ground truth.

Then, TEDS normalizes this distance into a similarity score between 0 and 1:

$$
TEDS = 1 - \frac{\text{EditDistance}(T_{pred}, T_{gt})}{\max(|T_{pred}|, |T_{gt}|)}
$$

So:

- TEDS = 1 → perfectly matched structure and content.
- TEDS = 0 → completely dissimilar.

In later benchmarks (like OmniDocBench, 2024), a simplified version called TEDS-S appears. It keeps the same idea but ignores minor text differences, focusing mostly on structural matching. It’s lighter to compute when content accuracy isn’t the main concern

### 2.3. Model and Result

**Model: Encoder–Dual-Decoder (EDD) architecture**
They propose a neural architecture that takes as input the image of the table, and produces the HTML markup with cell text.

**Components:**

1. **Encoder** — a CNN-based visual encoder (e.g. convolutional layers) that extracts image features from the input cropped table image.
2. **Structure decoder** — a recurrent / attention decoder that outputs structural tokens (e.g. HTML tags, <tr>, <td>, </td>, etc.). This decoder builds out the tree structure.
3. **Cell decoder** — triggered in tandem with structure decoder: when structure decoder emits a <td> (or a cell node), the cell decoder is invoked to output the textual content of that cell (token by token). The hidden state of the structure decoder may help the cell decoder focus on the correct image region.

The decoders interact — the structure decoder “drives” the generation of which cells exist in what order, and the cell decoder fills in content. This separation helps the model specialize in structure vs text.

They train the model end-to-end, minimizing a combined loss (structure + content).

**Baselines**

They compare EDD against prior methods (not necessarily deep models) or simpler encoder-decoder variants trained to map table images → LaTeX or HTML. They report results in terms of TEDS scores, and show gains in structural and content correctness.

**Quantitative results**

Some key **results**:

- Their EDD achieves a TEDS improvement of about **9.7 percentage** points absolute over the previous state-of-the-art.
- In ablation / error analysis, they find that the structure decoder helps the cell decoder focus better, reducing content errors, and that their approach is more robust to complex layouts (e.g. merged cells, irregular spans) than flat decoders.

The paper includes qualitative examples (cases the model handles well or fails) indicating that difficulties remain when the table is very messy, or when cell text is small / low contrast.

## 3. C**ISOL**

CISOL is focused on civil engineering / construction industry documents—specifically, it targets real-world documents such as “Steel Ordering Lists” (i.e. tables in structural engineering, ordering/quantity lists)

### 3.1 Dataset

The authors argue that many table recognition datasets are biased toward scientific, academic, or generic document types; but in specialized industries (construction, engineering), formats have domain‐specific quirks (e.g. embedded drawings, technical diagrams, irregular table layouts). CISOL fills that niche.

- CISOL consists of over 120,000 annotated table instances drawn from about800 document images (so many tables per image).
- The annotations include table detection (TD) and table structure recognition (TSR) tasks (i.e. recognizing where tables are, and the internal structure—rows, columns, spanning cells, header vs body).
- For each table, they annotate spanning cells, columns, rows, headers, etc. And likely cell boundaries (though I don’t see explicit mention of full content recognition beyond structure in the abstract).
- Because the documents include drawings, diagrams, background technical content, there are “distractor” elements. The variety in spanning (merged) cells is emphasized as a challenge.
- They design it to be open & extensible: one can extend or adapt it, or reuse annotation schema.

### 3.2. Metrics

- For table detection (TD), they use mean Average Precision (mAP) over IoU thresholds. In particular, they mention mAP@0.5:0.95:0.05 (i.e. the standard COCO‐style metric sweeping IoU thresholds from 0.5 to 0.95 in steps of 0.05).
- For table structure recognition (TSR), they compare structural outputs (rows, columns, spanning) against ground-truth annotations. The paper claims the benchmark shows that using YOLOv8 (for detection + recognition) achieves better results than a TSR-specialized model (TATR) on this domain.
- They also report annotation consistency / inter‐annotator agreement via a metric “K – α” (Kappa‐Alpha or some consistency score) to validate that their annotations are reliable.
- They analyze results over different IoU thresholds to see how sensitive detection is, and compare model convergence thresholds.

### 3.3. Models and Results

- Using YOLOv8 (a modern object detection / segmentation / detection architecture) on CISOL, they achieve 67.22 mAP@0.5:0.95:0.05 for the detection/recognition tasks. This is a headline benchmark.
- They compare that to TATR, a model more specialized for Table Structure Recognition (TSR). They find that YOLOv8 outperforms TATR in the CISOL domain. That suggests that a generic detection + recognition model, when tuned properly, can beat a domain-specialized structural model, at least in this domain.
- They present consistency / annotation quality: using the K–α metric, they show that the annotation agreement is good, which supports the validity of the dataset as a benchmark.
- They also include qualitative examples illustrating challenging cases: for instance, tables embedded within other tables, large variation in how spanning cells extend, and distractor elements (technical drawings, diagrams) complicating segmentation. In some cases, spanning cells stretch over embedded tables, or over irregular layouts.

## 4. TabRecSet

### 4.1. Dataset

The authors argue that most existing table recognition datasets are limited in domain (e.g. academic documents) and do not support end-to-end table recognition (i.e. doing table detection + structure recognition + content recognition in a unified pipeline). They aim to provide a more “in-the-wild” and multilingual benchmark.

- Size: about 38,100 table instances overall.
- Language breakdown: about 20,400 tables in English, about 17,700 in Chinese.
- Variety of forms / distortions: includes both border-complete and border-incomplete tables; regular and irregular tables (rotated, skewed, distorted)
- Diverse scenarios: the images are drawn from multiple capture modalities (scanned, camera-taken), document types (documents, financial invoices, test papers, Excel tables) and varying image quality.
- Annotations: They annotate, for each table instance, • The spatial polygon of the table body (rather than simple bounding box) — better for irregular shapes:
    - Cell-level spatial logical annotations (i.e. the layout / structure of the cells)
    - The text content for each cell (non-empty ones)

### 4.2. Metrics

Because the dataset supports end-to-end table recognition, the benchmark combines three subtasks:

1. Table Detection (TD) — locate the table region(s) in the image;
2. Table Structure Recognition (TSR) — figure out the cell structure / logical layout of the table;
3. Table Content Recognition (TCR) — read/recognize the text in each cell.

For metrics:

- For detection (TD), they presumably use standard object detection metrics (e.g. intersection-over-union, precision/recall) though the paper emphasizes using polygon annotations instead of boxes/quads, enabling more flexible shape matching.
- For structure + content, the evaluation must check both whether the cell layout is correct and whether the recognized text matches ground truth. The paper doesn’t deeply detail a new metric in the abstract, but given the lineage of table recognition works, it is likely they either adopt or adapt tree-edit-distance (TEDS) or HTML-based structural similarity metrics, augmented with content correctness. (They don’t state in abstract exactly, but the field precedent suggests this).
- Their dataset allows evaluation under “polygon-level” spatial matching, which gives them more fidelity in irregular table cases.

In sum: the benchmark is holistic — you’re judged not simply on detecting tables, or parsing their layout, or OCRing text in isolation — but on performing all three in synergy.

### 4.3. Models and Results

Because the paper is mainly dataset + benchmark paper, the results are more about baseline methods to set a reference than heavy model contributions.

- They evaluate several baseline models / pipelines (off-the-shelf or adapted) on TabRecSet (English and Chinese). (The paper in full likely shows a table of performance across the three subtasks).
- These baselines typically suffer significantly when moving from “clean / regular table images” to more challenging, distorted, or irregular ones (e.g. rotated, camera-captured). The more challenging the geometric distortions or irregular borders, the larger the drop in performance. (This is a qualitative conclusion drawn in the paper’s motivation).
- The baselines show that performance degrades more sharply in the Chinese tables than in English ones (likely due to script complexity, variation) — though exact numbers for that are in the paper.
- The results serve as a baseline (i.e. “you should beat this”) rather than state-of-the-art claims. The authors hope future models, especially those designed for end-to-end parsing under distortion, will improve on these baselines.

## 5. Proposal Benchmark Set

### **5.1. The Three Levels of Evaluation**

All document parsing problems decompose naturally into three evaluative levels:

1. **Detection** — where are the objects (tables, figures, text blocks)?
2. **Structure recognition** — how are they arranged (cells, rows, columns, reading order, hierarchy)?
3. **Content extraction** — what is written or drawn there (text, formulas, numbers, symbols)?

Each level has its own metric family.

### **5.2. Level 1: Detection Metric**

**5.2.1. Precision / Recall / F1** — classic overlap-based detection metrics.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$

\text{Recall} = \frac{TP}{TP + FN}

$$

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**5.2.2. mAP@IoU** thresholds — mean Average Precision over intersection-over-union (IoU) thresholds (e.g. 0.5:0.95:0.05, the COCO protocol).

Interpretation: these measure whether you found the right regions, not whether you understood their contents.

### **5.3. Level 2: Structural Metrics**

They judge how faithfully a model reconstructs the spatial and logical structure of a document or table.

**5.3.1. Tree-Edit-Distance-based Similarity (TEDS)**

$$
TEDS = 1 - \frac{\text{EditDistance}(T_{pred}, T_{gt})}{\max(|T_{pred}|, |T_{gt}|)}
$$

- Introduced by *PubTabNet* (2019).
- Represents HTML structure as a tree; measures how many insertions/deletions/replacements are needed to transform prediction → ground truth.
- Combines structure and content similarity.
- Range: 0 – 1 (higher is better).

**5.3.2. Reading-Order Edit Distance:** 

It’s just the Levenshtein (edit) distance between two sequences:

- One sequence = the ground-truth block order
- The other sequence = the predicted order
- Computes edit distance between predicted and true reading sequences of text blocks.
- Lower is better.

Each “edit” operation (insert, delete, swap/substitute) costs 1.

**5.3.3. Matching-based metrics (IoU for blocks / spans)**

- Some benchmarks (OmniDocBench) compute overlaps between predicted blocks and ground truth spans, aligning via minimal cost matching before scoring text accuracy.

Interpretation: structural metrics ask, did you rebuild the skeleton of the page correctly?

### **5.4. Level 3: Content Metrics**

Evaluate correctness of text, formulas, or table cell content.

**5.4.1. Levenshtein distance:** minimum number of single-character edits. 

**Example**: Transform `"kitten"` → `"sitting"`:

1. Substitute `'k'` → `'s'` → `"sitten"`
2. Substitute `'e'` → `'i'` → `"sittin"`
3. Insert `'g'` → `"sitting"`

→ Levenshtein Distance = 3

**Formula**

Let the two strings be:

$$
a = a_1a_2\dots a_m\\
b = b_1 b_2 \ldots b_n
$$

$$
D(i, j) =
\begin{cases}
0, & \text{if } i = 0 \text{ and } j = 0 \\
i, & \text{if } j = 0 \\
j, & \text{if } i = 0 \\
\min
\begin{cases}
D(i-1, j) + 1 \\[4pt]
D(i, j-1) + 1 \\[4pt]
D(i-1, j-1) + [a_i \neq b_j]
\end{cases}
, & \text{otherwise}
\end{cases}
$$

```python
for i in [0..len(a)]: D[i][0] = i
for j in [0..len(b)]: D[0][j] = j

for i in [1..len(a)]:
    for j in [1..len(b)]:
        cost = 0 if a[i-1] == b[j-1] else 1
        D[i][j] = min(
            D[i-1][j] + 1,      # deletion
            D[i][j-1] + 1,      # insertion
            D[i-1][j-1] + cost  # substitution
        )

return D[len(a)][len(b)]
```

**5.4.2. Normalized Edit Distance (NED):** Common in OmniDocBench for OCR quality.

$$
\text{NED} = 1 - \frac{\text{LevenshteinDistance}(pred, gt)}{\max(|pred|, |gt|)}

$$

**5.4.3. BLEU (Bilingual Evaluation Understudy)**: Measures *n-gram precision* between a machine translation (candidate) and one or more reference translations.

$$
{BLEU}_N = \operatorname{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \ln p_n\right)
$$

where:

$$

\operatorname{BP} =
\begin{cases}
1, & c > r \\[4pt]
\exp\!\left(1 - \dfrac{r}{c}\right), & c \le r
\end{cases}
$$

- $p_n$ : modified *n-gram* precision
- $w_n$: weights (usually $w_n = 1/N$)
- $c$: candidate length
- $r$: reference length
- $\operatorname{BP}$: brevity penalty (penalizes short translations)

**5.4.4. METEOR (Metric for Evaluation of Translation with Explicit ORdering):** Balances *precision* and *recall*, while accounting for stemming, synonyms, and word order (fragmentation).

$$
P = \frac{m}{c}, \quad R = \frac{m}{r}
$$

$$
F_{\text{mean}} = \frac{P \cdot R}{\alpha P + (1 - \alpha) R}
$$

$$
\operatorname{Pen} = \gamma \left(\frac{ch}{m}\right)^{\beta}
$$

$$
\mathrm{METEOR} = (1 - \operatorname{Pen}) \cdot F_{\text{mean}}

$$

- $m$: number of unigrams matched (possibly via stemming/synonyms)
- $c, r$: candidate and reference lengths
- $ch$: number of matched *chunks* (continuous sequences of words)
- $\alpha$: recall weight (commonly ~0.9)
- $\gamma, \beta$: penalty hyperparameters (e.g., $\gamma = 0.5, \beta = 3$)

Use **BLEU** for large-scale corpus evaluation — it’s stable and simple.

Use **METEOR** when you need finer granularity or language flexibility.

Combine with semantic metrics (e.g. **BERTScore**, **COMET**) for modern evaluation.

**5.4.5. Formula metrics (CDM)** — OmniDocBench’s Complex Document Matching metric for LaTeX formulas; tolerant to syntactic but not semantic variation.

![Overview of the Character Detection Matching (CDM), consisting of four main stages. **(1)** Element Localization, where bounding boxes of individual elements are extracted. **(2)** Element Region Matching, which employs a bipartite graph matching method to pair prediction with ground truth elements. **(3)** Invalid Match Elimination, where inconsistent matches are discarded through token and positional relationship checks. **(4)** Metric Calculation, where matching accuracy is evaluated using the F1-Score and ExpRate@CDM.](Benchmark%20and%20Evaluation/image.png)

Overview of the Character Detection Matching (CDM), consisting of four main stages. **(1)** Element Localization, where bounding boxes of individual elements are extracted. **(2)** Element Region Matching, which employs a bipartite graph matching method to pair prediction with ground truth elements. **(3)** Invalid Match Elimination, where inconsistent matches are discarded through token and positional relationship checks. **(4)** Metric Calculation, where matching accuracy is evaluated using the F1-Score and ExpRate@CDM.

### **5.5. Level 4: End-to-End Metrics**

End-to-end document parsing merges all previous aspects into one score.

**OmniDocBench Overall Score:** Balanced average of text, table, and formula accuracy.

$$
Overall = \frac{1}{3}\big[(1 - NED) + \text{TEDS} + \text{CDM}\big]
$$

**Weighted Fused Metrics**

Some works weight text, structure, and content differently per document type (e.g. more weight on layout for slides, on text for reports).

**Module-wise Breakdown**

Recent benchmarks (OmniDocBench, TabRecSet) publish module-specific and per-document-type scores to reveal weaknesses (e.g. “VLM models degrade on slides; pipelines fail on merged-cell tables”).

## Reference

1. [**OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations**](https://arxiv.org/pdf/2412.07626)
2. [**A large-scale dataset for end-to-end table recognition in the wild**](https://arxiv.org/abs/2303.14884)
3. [**CISOL: An Open and Extensible Dataset for Table Structure Recognition in the Construction Industry**](https://arxiv.org/html/2501.15469v1)
4. [**Image-based table recognition: data, model, and evaluation**](https://arxiv.org/abs/1911.10683)
5. [**CDM: A Reliable Metric for Fair and Accurate Formula Recognition Evaluation**](https://arxiv.org/html/2409.03643v1)

## About me

[tungedng2710 - Overview](https://github.com/tungedng2710)