# SpellSpark: A Spelling Growth Recommender System for Kids

## 📚 Project Overview

SpellSpark is a Machine Learning–powered spelling practice system designed to help children strengthen their spelling skills by learning underlying error patterns rather than memorizing isolated corrections.

The system uses a **teacher–student dual‑model architecture** that:

1. Classifies spelling error types
2. Generates compact numerical embeddings representing each mistake
3. Predicts similar error‑prone patterns for unseen words
4. Recommends practice words based on embedding similarity

This allows children to practice words that share similar spelling challenges, promoting pattern‑based learning.

---

## 🎯 The Problem & Solution

### The Challenge
Research shows that children tend to make predictable categories of spelling errors at different developmental stages. Traditional rule‑based tools struggle because they:

- Require manual rule creation
- Cannot generalize to new vocabulary
- Perform poorly on phonetic or semantic errors
- Do not learn from children’s mistakes

### Our Solution
SpellSpark replaces rules with machine-learned representations:

- The **Teacher Model** learns real mistake patterns
- The **Student Model** predicts mistake‑like embeddings for new words
- The system recommends similar words through embedding similarity
- The curriculum continuously improves as new user mistakes are collected

---

## 🏗️ System Architecture

### High-Level Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     SpellSpark System                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Vue.js Frontend (UI Layer)                    │   │
│  │  - Interactive spelling game                            │   │
│  │  - Audio pronunciation support                          │   │
│  │  - Visual feedback animations                           │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                      │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │      Backend API (Azure Function Proxy)                 │   │
│  │ - Forwards requests to Teacher Model endpoint           │   │
│  │ - Returns responses to UI                               │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                      │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │           ML Model Pipeline                             │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │          Teacher Model (BiLSTM)                  │   │   |
│  │  |  Input: (correct_word, misspelled_word)          │   │   │
│  |  |  Outputs:                                        │   │   │
│  |  |     1. Error Type (argmax of softmax)            │   │   │
│  |  |     2. 64‑dimensional mistake embedding          │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                       │                                 │   │
│  │  ┌────────────────────▼─────────────────────────────┐   │   │
│  │  │          Curriculum Embedding Store              │   │   │
│  |  |   - Precomputed 64‑dim embeddings for            |   |   |
|  |  |     curriculum words                             │   │   │
│  |  |   - Used for similarity search during inference  │   │   │
│  │  └────────────────────▲─────────────────────────────┘   │   │
│  │                       │                                 │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │          Student Model (CNN)                     │   │   │
│  │  |   Input: correct_word                            │   │   │
│  |  |   Output: 64‑dim embedding (teacher‑aligned)     │   │   │
│  │  |   Used only for batch‑generation of curriculum   │   │   │
│  │  |   embeddings during offline training             │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Models

### Model 1: Character-Level BiLSTM (Error Classifier)

**Purpose**: Given a pair (correct_word, misspelled_word), the teacher model produces:
- Error Type Classification
         - The softmax layer produces probabilities over error classes; the predicted error type is the argmax.

- 64‑Dimensional Mistake Embedding
         - The activated weights from the final hidden layer form a dense representation of the mistake pattern.
         - This embedding is used directly in similarity search.
 
**Architecture**:
- Character‑level encoding of both words
- Bidirectional LSTM layers
- Final hidden layer → 64‑dim embedding
- Softmax classifier in parallel for error type

**Why BiLSTM?**
- Captures sequential character dependencies in both forward and backward directions
- Learns rich semantic and phonetic representations of spelling patterns
- Generates meaningful embeddings that cluster similar error types together in vector space
- Outperforms rule-based systems on complex errors like phonetic substitutions

**Training Data**:
- Children’s essay spelling mistakes
- Common English misspellings
- Automatically labeled error categories

---

### Model 2: Student Model (Character‑Level CNN)

**Purpose**: Predict a **64‑dimensional embedding** for any correct word that mimics the teacher model’s embedding space.

**Architecture**:
- Input: correct word
- Convolution layer with 64 filters and kernel sizes [2, 3, 4]
- Max pooling
- Dense layers
- Output: 64‑dimensional embedding

**Why Character-Level CNN?**
- Convolutional filters naturally detect local patterns and character subsequences
- Efficient n-gram detection without manual feature engineering
- Significantly outperforms rule-based approaches on generalization tasks

**Role in System**
- Used offline
- Performs batch inference over the curriculum word list
- Stores embeddings in the Curriculum Embedding Store
- Does not interact with the Teacher Model directly

---

## 🔄 Workflow

### Offline Training Pipeline

```
Teacher Model trains on labeled (correct, misspelled) pairs
         │
         ▼
Teacher generates embeddings for all correct words
         │
         ▼
Student Model trains to mimic teacher embeddings
         │
         ▼                          
Student Model performs batch inference on curriculum words
         │
         ▼  
Store 64‑dim curriculum embeddings in the embedding store
```

### Runtime Inference (User Interaction)

```
User submits: (correct_word, misspelled_word)
         │
         ▼
Azure Function Proxy
         │
         ▼
Teacher Model
Outputs:
- error type
- 64‑dim mistake embedding
         │
         ▼
Similarity search against curriculum embedding store
         │
         ▼
Return top‑k similar curriculum words
         │
         ▼
UI behavior:
- Suggest the top-ranked similar word
- If the correct_word is NOT in top‑k: Save (correct_word, misspelled_word) as new training data for future improvements
```

---

## 🎮 UI/Frontend Layer

**Technology**: Vue 3 with Vite

**Location**: `spell-spark-ui/`

### Features:
- Gamified spelling practice
- Audio pronunciation
- Interactive particle system
- Dark theme
- Immediate positive/negative feedback

---

## 📊 Data Pipeline

### Training Data Sources

1. **KidSpell Essay Errors** (1,361 pairs)
   - Authentic spelling mistakes from children's essays
   - Labeled by error type using rule-based heuristics

2. **Wikipedia Misspellings** (2,455 pairs)
   - Common misspellings with known corrections
   - Broader coverage of error patterns

3. **ReadingVine Vocabulary Bank**
   - Curated list of age-appropriate vocabulary for children
   - Used as target words for recommendations

### Error Type Categories

Based on analysis of children's spelling patterns:
- **Double Letter Drops**: Omitting repeated consonants (necessary → necassary)
- **Vowel Substitutions**: Incorrect vowel selection
- **Consonant Substitutions**: Wrong consonant usage
- **Phonetic Substitutions**: Sound-based errors
- **Letter Transpositions**: Swapped character positions
- **Extra/Missing Letters**: Insertion or deletion of characters
- [Additional categories based on dataset analysis]

---

## 📚 References

- Niolaki, S., et al. (2023). "Developmental differences in spelling error patterns." *Journal of Learning Disabilities*.
- Downs, K., et al. (2020). "KidSpell: A dataset and system for spelling error analysis in children's writing." *NLP in Education Workshop*.
- ReadingVine. (n.d.). Vocabulary words for children. Retrieved from https://www.readingvine.com/

---

**Last Updated**: 2026-08-12
