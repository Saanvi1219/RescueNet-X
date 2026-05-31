# RescueNet-X

## AI-Powered Missing Child Rescue System using Face Recognition, Age-Invariant Learning, Trajectory Intelligence, Graph Neural Networks, and Multimodal Fusion

### Overview

RescueNet-X is a multimodal AI framework designed to assist in missing child identification and rescue prioritization. The system integrates facial recognition, age-invariant matching, trajectory intelligence, graph-based reasoning, and multimodal fusion to generate rescue recommendations and confidence scores.

The framework combines information from multiple modalities to improve search efficiency and rescue decision-making.

---

## Key Features

### Face Recognition Module

* Built using UTKFace dataset
* Learns facial embeddings for identity representation
* Generates face similarity scores

### Age-Invariant Matching Module

* Built using FG-NET dataset
* Siamese learning architecture
* Handles age progression and appearance changes
* Produces age-invariant confidence scores

### Trajectory Intelligence Module

* Built using Microsoft GeoLife GPS trajectories
* LSTM-based temporal modeling
* Processes long-range movement patterns
* Achieved 91.35% trajectory prediction accuracy

### Graph Neural Network Module

* Multi-Graph GNN built from individual trajectory graphs
* Models spatial relationships between movement patterns and rescue zones
* Generates graph-based rescue zone rankings

### Multimodal Fusion Module

Combines:

* Face Similarity Score
* Age-Invariant Similarity Score
* Trajectory Confidence Score
* Graph Reasoning Score

to produce a final Rescue Confidence Score.

---

## Datasets

### UTKFace

* 23,000+ facial images
* Used for identity representation learning

### FG-NET Aging Dataset

* 82 identities
* Age progression learning

### Microsoft GeoLife

* 18,670 trajectories
* More than 12 million GPS observations

---

## Architecture

Missing Child Case

↓

Face Encoder (UTKFace)

↓

Age-Invariant Matching (FG-NET)

↓

Trajectory Intelligence (GeoLife + LSTM)

↓

Graph Neural Network

↓

Multimodal Fusion

↓

Final Rescue Confidence

↓

Top Rescue Zone Recommendations

---

## Results

| Module            | Result                        |
| ----------------- | ----------------------------- |
| Face Encoder      | Identity Embeddings Generated |
| Age Module        | Age-Invariant Matching        |
| Trajectory LSTM   | 91.35% Accuracy               |
| Multi-Graph GNN   | Top-3 Rescue Zone Ranking     |
| Multimodal Fusion | Final Rescue Confidence Score |

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* NumPy
* Pandas
* Scikit-Learn
* NetworkX
* Matplotlib

---

## Future Work

* Real-time CCTV integration
* Transformer-based trajectory modeling
* Large-scale heterogeneous graph reasoning
* Explainable AI for rescue recommendations
* Deployment as a web application

---

## Authors

Developed as a research-oriented multimodal AI system for missing child rescue and search prioritization.
