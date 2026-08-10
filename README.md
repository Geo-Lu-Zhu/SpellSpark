# SpellSpark: A Spelling Growth Recommender System for Kids

## 📚 Project Overview

SpellSpark is an innovative Machine Learning-powered application designed to help children practice spelling more effectively. Unlike traditional rule-based spell checkers, SpellSpark uses a dual-model neural network architecture to:

1. **Classify spelling errors** by their underlying type (e.g., phonetic substitution, double-letter drops)
2. **Predict error patterns** for new vocabulary words
3. **Recommend similar words** that share the same spelling challenges

This dynamic approach shifts the paradigm from memorizing isolated instances to practicing and mastering underlying spelling patterns.

---

## 🎯 The Problem & Solution

### The Challenge
Research shows that children make distinct types of spelling errors at different developmental stages (Niolaki et al., 2023). Current state-of-the-art tools like KidSpell rely on programmed, rule-based algorithms that:
- Require manual hard-coding for every possible mistake
- Cannot generalize across vocabulary
- Fail to understand complex semantic and phonetic errors
- Cannot learn from real-world user data

### Our Solution
SpellSpark replaces static rules with **machine learning-driven intelligence**:
- Learns from both error pairs and structural patterns in correct words
- Automatically discovers and adapts to spelling error categories
- Provides intelligent recommendations for personalized practice
- Enables continuous improvement through real-world user feedback

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SpellSpark System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Vue.js Frontend (UI Layer)                    │   │
│  │  - Interactive spelling game with visual feedback       │   │
│  │  - Audio pronunciation support                          │   │
│  │  - Real-time particle animations                        │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │      Backend API (Flask/Python)                         │   │
│  │  - Processes user spelling submissions                  │   │
│  │  - Interfaces with ML models                            │   │
│  │  - Generates recommendations                            │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │           ML Model Pipeline                             │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │ Model 1: Character-Level BiLSTM                 │   │   │
│  │  │ (Error Classifier)                              │   │   │
│  │  │ Input: (correct_word, misspelling) pairs        │   │   │
│  │  │ Output: Error Profile (probability distribution)│   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │                       │                                  │   │
│  │  ┌────────────────────▼──────────────────────────────┐  │   │
│  │  │ Model 2: Character-Level CNN                     │  │   │
│  │  │ (Error Predictor)                                │  │   │
│  │  │ Input: correct_word only                         │  │   │
│  │  │ Output: Predicted Error Profile                  │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                       │                                  │   │
│  │  ┌────────────────────▼──────────────────────────────┐  │   │
│  │  │ Vocabulary Bank Database                         │  │   │
│  │  │ (Pre-computed Error Profiles for all words)      │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Models

### Model 1: Character-Level BiLSTM (Error Classifier)

**Purpose**: Classify the type of spelling error given a correct-misspelled word pair.

**Architecture**:
- **Input Layer**: Character-level encoding of (correct_word, misspelled_word) pairs
- **BiLSTM Layers**: Bidirectional Long Short-Term Memory networks capture context in both directions
- **Hidden Layer**: Last hidden layer serves as a rich latent representation (embedding) of the error pattern
- **Output Layer**: Softmax classifier over multiple error type categories

**Why BiLSTM?**
- Captures sequential character dependencies in both forward and backward directions
- Learns rich semantic and phonetic representations of spelling patterns
- Generates meaningful embeddings that cluster similar error types together in vector space
- Outperforms rule-based systems on complex errors like phonetic substitutions

**Key Outputs**:
1. **Error Profile**: A probability distribution over error classes (e.g., 50% Double Letter Drop, 30% Phonetic Substitution, 20% Vowel Substitution)
2. **Embeddings**: Latent representations from the last hidden layer used for similarity comparison

**Training Data**:
- 1,361 (correct, misspelled) pairs from children's essays (KidSpell dataset)
- 2,455 misspellings from Wikipedia common misspellings corpus
- Automatically labeled with error types using edit distance and rule-based heuristics

**Performance** (Proof of Concept):
- Accuracy: 70% (learning from scratch without hand-engineered features)
- Successfully distinguishes between structural errors (missing letters) and complex errors (phonetic substitutions)

---

### Model 2: Character-Level CNN (Error Predictor)

**Purpose**: Predict the likely error profile for any correctly spelled word, enabling recommendations for unseen vocabulary.

**Architecture**:
- **Input Layer**: Character-level encoding of the correct word
- **Convolutional Layers**: Multiple parallel filters act as n-gram detectors to identify tricky patterns (double consonants, vowel clusters, etc.)
- **Pooling Layers**: Extract the most salient features from each n-gram window
- **Dense Layers**: Map learned patterns to error profiles
- **Output Layer**: Predicts a continuous probability distribution (error profile) for each word

**Why Character-Level CNN?**
- Convolutional filters naturally detect local patterns and character subsequences
- Efficient n-gram detection without manual feature engineering
- Can be trained to predict continuous probability distributions (not just discrete classes)
- Significantly outperforms rule-based approaches on generalization tasks

**Key Outputs**:
- **Predicted Error Profile**: A probability distribution over error types for any vocabulary word
- **Similarity Scores**: Used to rank and recommend words with matching error patterns

**Training Data**:
- Correct words from the KidSpell and Wikipedia datasets
- Target labels: Error profiles generated by Model 1
- Enables Model 2 to learn how structural patterns in words predict their vulnerability to specific error types

**Performance** (Proof of Concept):
- Average Cosine Similarity: 0.67 (vs. 0.49 for KNN baseline)
- Successfully generalizes error patterns to unseen vocabulary words
- Captures nuanced relationships between word structure and error susceptibility

---

### Performance Comparison: Neural Networks vs. Rule-Based Baseline

| Aspect | KNN Baseline | BiLSTM (Model 1) | CNN (Model 2) |
|--------|-------------|------------------|---------------|
| **Model 1 Accuracy** | 90.6% | 70% | - |
| **Model 2 F1-Score** | Lower across classes | - | Higher across classes |
| **Error Profile Cosine Similarity** | 0.49 | - | 0.67 |
| **Handles Phonetic Errors** | ❌ Struggles | ✅ Learns naturally | ✅ Learns patterns |
| **Generalizes to New Words** | ❌ Limited | - | ✅ Excellent |
| **Interpretability** | ✅ Highly transparent | ⚠️ Black-box | ⚠️ Black-box |
| **Data Efficiency** | ✅ Works with small data | ⚠️ Requires more data | ⚠️ Requires more data |

**Key Insight**: While KNN achieves 90.6% accuracy through hand-crafted features that "cheat" on structural errors, neural networks excel at understanding complex patterns and generalizing to unseen vocabulary—which is essential for SpellSpark's real-world mission.

---

## 🔄 End-to-End Workflow

### Phase 1: Training (Offline, One-time Setup)

```
Raw Datasets (KidSpell, Wikipedia)
         │
         ▼
Error Classification (Edit Distance + Rules)
         │
         ▼
(Correct, Misspelled, Error_Type) Training Data
         │
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     │
Train Model 1: BiLSTM                         │
         │                                     │
         ▼                                     │
Generate Error Profiles via Model 1           │
(for all correct words in training data)      │
         │                                     │
         ├─────────────────────────────────────┘
         │
         ▼
(Correct_Word, Error_Profile) Training Data
         │
         ▼
Train Model 2: CNN
         │
         ▼
Generate Vocabulary Bank
(Each word paired with predicted error profiles)
```

### Phase 2: Inference (Runtime, Per User Interaction)

```
User Input: Spelling Attempt
         │
         ├─ (Correct Word, User Misspelling)
         │
         ▼
Model 1 (BiLSTM) Inference
         │
         ▼
Error Profile (user's current mistake type)
         │
         ├─ Query Vocabulary Bank
         │
         ▼
Rank words by profile similarity (cosine distance)
         │
         ▼
Return Top-K Similar Words for Practice
         │
         ▼
Display recommendations to user
```

### Phase 3: Continuous Learning (MLOps Feedback Loop)

```
Every User Interaction:
         │
         ├─ Capture (correct_word, misspelling) pair
         │
         ├─ Store in training buffer
         │
         ├─ (Optional) Trigger retraining pipeline when buffer reaches threshold
         │
         └─ Update Model 1 and regenerate Vocabulary Bank
         
Result: Models continuously improve as they learn from real children's mistakes
```

---

## 🎮 UI/Frontend Layer

**Technology**: Vue 3 with Vite

**Location**: `spell-spark-ui/`

### Key Components:

1. **Game Interface**
   - Interactive spelling practice in a gamified environment
   - Audio pronunciation support using Web Speech API
   - Real-time visual feedback through particle animations

2. **User Interaction Flow**
   - Display word definition
   - Play audio pronunciation
   - User types spelling attempt
   - Submit via Enter key
   - Visual feedback:
     - ✅ Correct: Glowing fire animation + applause sound
     - ❌ Incorrect: Smoke fade animation + error sound

3. **Particle System**
   - Canvas-based animated particles spawn when user types
   - Creates engaging visual feedback and encourages interaction
   - Performance-optimized with requestAnimationFrame

### Features:
- Dark theme optimized for children
- Responsive design
- Accessibility: Text-to-speech for pronunciations
- Sound effects for positive/negative reinforcement

---

## 🔗 Backend API Layer

**Technology**: Python with Flask (or equivalent)

**Responsibilities**:

1. **Model Serving**
   - Load pre-trained Model 1 (BiLSTM) and Model 2 (CNN)
   - Provide inference endpoints for error classification and recommendation

2. **Request Handling**
   ```
   POST /api/check-spelling
   {
     "correct_word": "necessary",
     "user_misspelling": "necassary"
   }
   
   Response:
   {
     "error_profile": {
       "double_letter_drop": 0.75,
       "vowel_substitution": 0.15,
       "consonant_substitution": 0.10
     },
     "recommended_words": [
       "Mississippi",
       "accommodate",
       "possess",
       ...
     ]
   }
   ```

3. **Data Collection & Logging**
   - Record every user interaction for MLOps pipeline
   - Maintain audit trail for model improvement

4. **Vocabulary Management**
   - Serve curated vocabulary bank
   - Return word definitions and pronunciation guidance

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

## 🚀 Getting Started

### Prerequisites
- Node.js >= 22.18.0
- Python 3.8+
- pip package manager

### Frontend Setup

```bash
cd spell-spark-ui
npm install
npm run dev              # Development server
npm run build            # Production build
```

### Backend Setup (Placeholder)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python app.py

# The API should run on http://localhost:5000
```

### Running the Application

1. Start the backend API
2. Start the Vue development server
3. Open browser to `http://localhost:5173` (or configured Vite port)
4. Click "Start Game" to begin practicing

---

## 📈 Expected Performance Improvements

### From POC to Production

The current proof-of-concept has not undergone hyperparameter tuning or architecture optimization. Expected improvements:

- **Model 1 (BiLSTM)**
  - Current: 70% accuracy
  - Expected with tuning: Closer to 80-85% (narrowing gap to KNN baseline)
  - Reason: Better optimization and more focused error classification

- **Model 2 (CNN)**
  - Current: 0.67 cosine similarity on error profiles
  - Expected with tuning: 0.75+ cosine similarity
  - Reason: Improved generalization to unseen vocabulary words

---

## 🔮 Future Roadmap

1. **Model Enhancements**
   - Hyperparameter optimization and architecture search
   - Integration of transfer learning from pre-trained language models
   - Multi-language support

2. **Personalization**
   - User-specific difficulty levels
   - Adaptive recommendation based on user history
   - Learning progress tracking

3. **Analytics & Insights**
   - Dashboard for educators to track class progress
   - Detailed error pattern analysis per student
   - Identify emerging error types in real-time

4. **Continuous Learning**
   - Automated retraining pipeline
   - A/B testing for model improvements
   - User feedback integration

---

## 📚 References

- Niolaki, S., et al. (2023). "Developmental differences in spelling error patterns." *Journal of Learning Disabilities*.
- Downs, K., et al. (2020). "KidSpell: A dataset and system for spelling error analysis in children's writing." *NLP in Education Workshop*.
- ReadingVine. (n.d.). Vocabulary words for children. Retrieved from https://www.readingvine.com/

---

## 👥 Project Team

Developed as a machine learning project focused on educational technology and AI for social good.

---

## 📄 License

[Add your license information here]

---

## 🤝 Contributing

Contributions are welcome! Please refer to the main project documentation for guidelines on:
- Adding new error type categories
- Improving model architecture
- Expanding vocabulary bank
- Enhancing UI/UX

---

## 📝 Updates

| Date | Update |
|------|--------|
| August 06, 2026 | Changed the number of training epochs to 50 to improve Model 1 performance; weighted avg macro F1 increased to 0.675. |
| August 06, 2026 | Modified data preprocessing for Model 1 so that all sequences are padded to the same length (max length = 55; informed by the length of the training/vocab data). The same max padding length is applied to all inputs during inference to ensure consistent and stable predictions. |
| August 04, 2026 | Switched to using the activated values from the last hidden layer (dim = 64) of Model 1 (BiLSTM) as the "error profiles" instead of the softmax label probabilities. |

---

**Last Updated**: 2026-08-06
