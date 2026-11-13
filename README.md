# Document Retrieval Using Cosine Similarity and SVD

## 📌 Project Overview

This project explores and compares different approaches to **information retrieval (IR)** using vector space models and matrix decomposition techniques.

The objective is to rank documents according to their relevance to a given query by comparing three methods:

1. **Basic Cosine Similarity** – Direct similarity computation between query and document vectors.
2. **Singular Value Decomposition (SVD)** – Dimensionality reduction to identify latent semantic relationships between terms and documents.
3. **Bidiagonalization + QR Iterations** – An alternative approach to approximating SVD through Householder transformations and Givens rotations.

The project also investigates reconstruction errors, the impact of dimensionality reduction, semantic relationships, and computational performance.

## 🎯 Objectives

- Implement cosine similarity for document ranking.
- Implement SVD manually using eigenvalue decomposition.
- Perform matrix bidiagonalization using Householder transformations.
- Implement QR iterations using Givens rotations.
- Compare document relevance scores across different methods.
- Analyze the effect of the reduced rank \(k\) on retrieval results.
- Study ambiguity and polysemy in document retrieval.
- Benchmark the execution time of the different approaches.
- Process custom document collections from TXT files.

## 🧮 Methodology

### 1. Document Representation

Documents are represented using a **Term-Document Matrix** \(D\), where:

- Rows represent unique terms.
- Columns represent documents.
- Each entry indicates whether a term appears in a document (binary representation).

A query is represented as a vector \(q\) in the same term space.

### 2. Cosine Similarity

The similarity between a query and a document is calculated using:

\[
\operatorname{sim}(q,d)=
\frac{q\cdot d}{\|q\|\|d\|}
\]

This approach provides a baseline for ranking documents based on their direct term overlap.

### 3. Manual SVD

The matrix is decomposed as:

\[
D = U\Sigma V^T
\]

The implementation uses eigenvalue decomposition of \(D^TD\) to calculate the singular values and corresponding singular vectors.

A reduced-rank approximation is then constructed:

\[
D_k = U_k\Sigma_k V_k^T
\]

This reduces dimensionality while retaining the most significant components of the original matrix.

### 4. Bidiagonalization + QR

An alternative SVD-oriented approach is implemented through two main stages:

- **Bidiagonalization:** Householder transformations are used to reduce the matrix to bidiagonal form.
- **QR Iterations:** Givens rotations are used to perform QR transformations iteratively until convergence.

This approach provides an opportunity to explore the numerical techniques underlying singular value decomposition.

### 5. Retrieval and Ranking

For each method, similarity scores are calculated between the query and all documents.

Documents can then be ranked in descending order of their relevance scores.

## 🧪 Experiments

The project includes several experiments.

### Experiment 1: Basic Document Retrieval

A small collection of mathematical documents is used to compare the three retrieval approaches for a query related to linear algebra and matrices.

### Experiment 2: Semantic Retrieval and Polysemy

A collection of economics and geology documents is used to investigate how dimensionality reduction influences retrieval.

Two example queries are considered:

- `dépression, croissance`
- `bassin, fiscalité`

The experiments examine:

- Document ranking for different values of \(k\).
- Reconstruction error as a function of the reduced rank.
- The impact of polysemy, particularly the term *dépression*, which can refer to different concepts depending on its context.

### Experiment 3: Performance Benchmarking

The execution time of the three methods is measured for different document collection sizes.

| Parameter | Values |
|---|---|
| Number of documents (\(N_d\)) | 5, 10, 15, 20, 50, 100, 200 |
| Number of terms (\(N_t\)) | \(3 \times N_d\) |
| Reduced rank (\(k\)) | 5 |
| Number of repetitions | 3 |

The benchmark generates a graph comparing the execution times of the three methods.

The resulting plot is saved as `benchmark_results.png`.

### Experiment 4: Custom TXT Document Processing

The project supports processing external document collections from a text file named `documents.txt`.

Each line represents one document, with its identifier followed by its terms.

Example:

```text
doc1 croissance PIB investissement
doc2 inflation monnaie depression
doc3 commerce exportation croissance
doc4 emploi chomage salaires
```

The program:

1. Reads and parses the document collection.
2. Extracts and sorts unique terms.
3. Constructs the term-document matrix.
4. Selects a query term randomly.
5. Computes relevance scores using all three methods.
6. Displays a comparison table of the results.

## 🛠️ Technologies Used

- **Python 3**
- **NumPy** – Matrix operations, vector calculations, and numerical linear algebra.
- **Matplotlib** – Data visualization and performance benchmarking.

## 📂 Project Structure

```text
Document-Retrieval/
│
├── main.py                  # Main implementation
├── documents.txt            # Custom document collection
├── benchmark_results.png    # Generated benchmark graph
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Execution

### 1. Clone the repository

```bash
git clone <repository-url>
cd Document-Retrieval
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy matplotlib
```

### 3. Prepare the document collection

Create a `documents.txt` file in the project directory following the format described above.

### 4. Run the project

```bash
python main.py
```

The program executes the predefined retrieval examples, reconstruction error analysis, performance benchmark, and custom TXT document processing.

## 📊 Expected Outputs

- Similarity scores for each retrieval method.
- Ranked document lists for different reduced ranks.
- Reconstruction error visualization.
- Performance comparison graph (`benchmark_results.png`).
- Comparison table for custom document collections.

## 📚 Key Concepts

This project provides practical experience with:

- Information Retrieval and Vector Space Models.
- Cosine Similarity.
- Linear Algebra and Matrix Decomposition.
- Singular Value Decomposition (SVD).
- Dimensionality Reduction.
- Householder Transformations.
- Givens Rotations and QR Algorithms.
- Latent Semantic Analysis (LSA).
- Polysemy and Semantic Relationships.
- Numerical Performance Analysis.

## 👥 Contributors

Developed as part of an academic project exploring mathematical techniques and algorithms for information retrieval.
