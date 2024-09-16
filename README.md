# Advanced Music Metadata Matching

This project leverages Natural Language Processing (NLP) for semantic search in the music industry, addressing the challenge of identifying and linking identical musical compositions across varying formats and spellings.

## Project Overview

I've developed a cutting-edge semantic matching system for music metadata, demonstrating proficiency in data science, machine learning, and software engineering principles.

### Key Achievements

- Improved model accuracy from 40% to 97% through iterative development and fine-tuning
- Implemented a robust data cleaning pipeline for a real-world, inconsistent dataset
- Successfully applied state-of-the-art NLP models (BERT, Sentence-BERT) to a domain-specific problem
- Achieved high accuracy (97%) with minimal fine-tuning data (1,000 samples), showcasing efficient domain adaptation

## Repository Structure

```
.
├── .gitignore
├── data-preparation.ipynb
├── data_cleaning.py
├── initial-pre-processing.ipynb
├── pipeline.py
├── README.md
└── semantic-poc.ipynb
```

### File Descriptions

- `data-preparation.ipynb`: Jupyter notebook for data preparation steps
- `data_cleaning.py`: Python script containing functions for data cleaning and character decoding
- `initial-pre-processing.ipynb`: Initial data preprocessing
- `pipeline.py`: Main pipeline for data processing and model execution
- `semantic-poc.ipynb`: Jupyter notebook demonstrating the proof of concept for semantic matching (incl. fine-tuning)

## Technical Skills Showcased

- Programming: Python, Pandas, NumPy
- Machine Learning: PyTorch, BERT, Sentence-BERT, FAISS
- Data Analysis: Exploratory Data Analysis (EDA), data cleaning, feature engineering
- Development: Jupyter Notebooks, VSCode, conda, modular code design

## Project Highlights

### Sophisticated Data Preparation

- Conducted thorough EDA to uncover and address data corruption and inconsistencies
- Implemented dynamic encoding to handle multiple character encodings within the dataset
- Developed flexible, reusable cleanup functions for efficient data processing

### Advanced Model Implementation and Optimisation

- Initial implementation: BERT Base Uncased (40% accuracy)
- Improved model: Sentence-BERT (88% accuracy)
- Fine-tuned model: Customised Sentence-BERT (97% accuracy)

### Innovative Problem-Solving Approaches

- Fail Fast Methodology: Rapidly implemented PoC to gather practical feedback and iterate quickly
- Strategic Framework Selection: Chose PyTorch for PoC flexibility, with modular design for potential transition to TensorFlow in production
- Efficient Fine-Tuning Strategy: Utilised specialised loss function for fine-tuning with only positive pairs, achieving significant accuracy boost with minimal data

## Note on Reproducibility

This repository is not intended for reproduction of the project. It serves as a demonstration of the work done and the approaches used in developing the music metadata matching system.

## Commit History

This is a copy of a private repository and doesn't contain commit history. Please email for access to the private repository.

## Future Enhancements

- Implement robust data version control for organised training experimentation
- Analyse false match similarity score distribution to understand unexpected central tendency around 60%
- Conduct more rigorous training, evaluation and testing with proper stratification to further improve accuracy
- Develop and deploy a web application prototype to demonstrate real-world applicability of the solution
- Implement secondary applications of the model, such as automated duplicate entry detection in large music catalogues

## Contact

[Email](mailto:archie.dobiss@gmail.com) | [LinkedIn](https://www.linkedin.com/in/adobiss)