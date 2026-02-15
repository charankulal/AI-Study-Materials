# Neural Network Architectures

## Table of Contents

- [Introduction](#introduction)
- [Feed Forward Neural Networks (FFNN)](#feed-forward-neural-networks-ffnn)
- [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
- [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
- [Transformer Architecture](#transformer-architecture)
- [Architecture Comparison](#architecture-comparison)
- [Choosing the Right Architecture](#choosing-the-right-architecture)

## Introduction

Different neural network architectures are designed to solve different types of problems. Just as you wouldn't use a hammer to cut wood, you shouldn't use a CNN for time series prediction or an RNN for image classification. Understanding the strengths and use cases of each architecture is crucial for successful deep learning applications.

### Why Different Architectures?

Each architecture has specialized design elements that make it particularly effective for certain types of data:

- **Spatial relationships** (images) → CNN
- **Sequential/temporal patterns** (text, time series) → RNN/Transformer
- **Independent data points** (tabular data) → FFNN
- **Long-range dependencies** (long texts, complex sequences) → Transformer

## Feed Forward Neural Networks (FFNN)

### What is FFNN?

The simplest and most straightforward neural network architecture where information flows in only one direction—from input layer through hidden layers to output layer. No loops, no feedback, no cycles.

### Architecture Structure

```
Input Layer (features)
      ↓
Hidden Layer 1 (neurons with weights and activations)
      ↓
Hidden Layer 2 (neurons with weights and activations)
      ↓
      ...
      ↓
Output Layer (predictions/classifications)
```

### Key Characteristics

**Unidirectional Flow**:

- Information moves forward only
- No connections back to previous layers
- No connections between neurons in the same layer

**Fully Connected** (Dense):

- Each neuron in one layer connects to all neurons in the next layer
- Also called "Dense layers" or "Fully Connected layers"

**Independent Processing**:

- Each input processed independently
- No memory of previous inputs
- No awareness of data order or sequence

### How It Works

**Step-by-Step Process**:

1. **Input**: Features enter the input layer
2. **Layer 1**: Each neuron calculates weighted sum + bias, applies activation
3. **Layer 2**: Takes Layer 1 outputs, repeats calculation
4. **Continue** through all hidden layers
5. **Output**: Final layer produces prediction

**Mathematical Operations** (simplified):

```
Layer output = activation(weights × inputs + bias)
```

### Use Cases

**Best For**:

- ✅ Tabular/structured data (spreadsheets, databases)
- ✅ Simple classification tasks
- ✅ Regression problems
- ✅ Feature-based predictions
- ✅ Independent data points

**Examples**:

1. **Predicting House Prices**
   - Input: Square footage, bedrooms, location, age
   - Output: Price prediction

2. **Credit Scoring**
   - Input: Income, debt, payment history, employment
   - Output: Credit risk category

3. **Medical Diagnosis (Tabular Data)**
   - Input: Age, blood pressure, cholesterol, test results
   - Output: Disease probability

4. **Customer Churn Prediction**
   - Input: Usage patterns, demographics, billing history
   - Output: Likelihood to cancel subscription

### Advantages

- ✅ Simple and intuitive
- ✅ Fast to train
- ✅ Works well for structured data
- ✅ Good starting point for most problems
- ✅ Easy to understand and implement

### Limitations

- ❌ No spatial awareness (not good for images)
- ❌ No temporal awareness (not good for sequences)
- ❌ Cannot handle variable-length inputs
- ❌ Large number of parameters for high-dimensional input (e.g., images)
- ❌ No built-in feature hierarchy learning

## Convolutional Neural Networks (CNN)

### What is CNN?

Specialized neural networks designed for processing grid-like data, particularly images. Uses convolutional operations to automatically detect spatial hierarchies of features.

### Key Innovation: Convolution

**Traditional FFNN Problem with Images**:

- Image 28×28 pixels = 784 inputs
- Image 224×224 pixels = 50,176 inputs
- Fully connected layer would need millions of parameters
- Doesn't capture spatial relationships

**CNN Solution**:

- Use small filters (kernels) that slide across the image
- Share weights across the image (same filter applied everywhere)
- Dramatically reduce parameters
- Automatically learn spatial features

### Architecture Components

#### 1. Convolutional Layers

**How They Work**:

- Small filter (e.g., 3×3 or 5×5) slides across image
- At each position, performs element-wise multiplication and sum
- Produces feature map highlighting specific patterns
- Multiple filters learn different features

**What They Detect**:

- **Early layers**: Edges, corners, colors, simple textures
- **Middle layers**: Shapes, patterns, object parts
- **Deep layers**: Complete objects, complex patterns

**Example**:

```
Input: 28×28 image
Filter: 3×3 (learns to detect edges)
Output: 26×26 feature map

Multiple filters → Multiple feature maps
32 filters → 32 feature maps (each detecting different patterns)
```

#### 2. Pooling Layers

**Purpose**: Reduce spatial dimensions, make features more robust

**Types**:

- **Max Pooling**: Takes maximum value in each region
- **Average Pooling**: Takes average value in each region

**Benefits**:

- Reduces computation
- Provides translation invariance (object can move slightly)
- Prevents overfitting
- Focuses on most important features

**Example**:

```
Input: 26×26 feature map
Max Pooling: 2×2 with stride 2
Output: 13×13 feature map (4× reduction)
```

#### 3. Fully Connected Layers

**Purpose**: Final classification after feature extraction

**Location**: At the end of the network

**Function**: Combines all learned features to make final prediction

### Typical CNN Architecture

```
Input Image (e.g., 224×224×3)
      ↓
Conv Layer 1 + ReLU (learns edges)
      ↓
Pooling Layer 1 (reduce size)
      ↓
Conv Layer 2 + ReLU (learns shapes)
      ↓
Pooling Layer 2 (reduce size)
      ↓
Conv Layer 3 + ReLU (learns objects)
      ↓
Pooling Layer 3 (reduce size)
      ↓
Flatten (convert to 1D)
      ↓
Fully Connected Layer (combine features)
      ↓
Output Layer (classifications)
```

### Use Cases

**Best For**:

- ✅ Image classification
- ✅ Object detection
- ✅ Image segmentation
- ✅ Facial recognition
- ✅ Medical image analysis
- ✅ Any grid-like spatial data

**Examples**:

1. **Image Classification**
   - Input: Photo
   - Output: "Cat", "Dog", "Bird"
   - Application: Photo organization

2. **Medical Image Analysis**
   - Input: X-ray, MRI, CT scan
   - Output: Disease detection, tumor localization
   - Application: Diagnostic assistance

3. **Autonomous Vehicles**
   - Input: Camera feed
   - Output: Pedestrians, vehicles, traffic signs, lanes
   - Application: Self-driving cars

4. **Facial Recognition**
   - Input: Face image
   - Output: Identity or verification
   - Application: Security, photo tagging

5. **Document Analysis**
   - Input: Scanned document
   - Output: Text regions, layout structure
   - Application: OCR, document processing

### Famous CNN Architectures

**LeNet-5** (1998): First successful CNN, digit recognition

**AlexNet** (2012): Breakthrough in ImageNet competition, popularized deep CNNs

**VGG** (2014): Very deep networks with small filters

**ResNet** (2015): Introduced skip connections, enabled very deep networks (150+ layers)

**Inception/GoogLeNet** (2014): Multi-scale feature processing

**EfficientNet** (2019): Optimized balance of depth, width, and resolution

### Advantages

- ✅ Excellent for images and spatial data
- ✅ Learns hierarchical features automatically
- ✅ Parameter sharing reduces model size
- ✅ Translation invariant (object can appear anywhere)
- ✅ Captures local patterns effectively

### Limitations

- ❌ Not designed for sequential data
- ❌ Fixed input size (though can be adapted)
- ❌ Computationally expensive for large images
- ❌ Requires significant training data for best performance

## Recurrent Neural Networks (RNN)

### What is RNN?

Neural networks with loops that allow information to persist, creating a form of "memory". Unlike FFNNs where each input is processed independently, RNNs maintain state and consider previous inputs when processing new ones.

### Key Innovation: Memory and Feedback

**The Problem with FFNN for Sequences**:

- Can't remember previous inputs
- Can't understand context or order
- Treats "dog bit man" same as "man bit dog"

**RNN Solution**:

- Maintains hidden state (memory)
- Hidden state passed to next time step
- Can remember and use previous information

### How RNNs Work

**Sequential Processing**:

```
Time step 1: Input word "The" → Hidden State 1
Time step 2: Input word "cat" + Hidden State 1 → Hidden State 2
Time step 3: Input word "sat" + Hidden State 2 → Hidden State 3
Time step 4: Input word "down" + Hidden State 3 → Output

Hidden state carries information from all previous time steps
```

**Feedback Loop**:

- Output at each step feeds back as input for next step
- Creates an internal "memory" of sequence
- Same weights used at each time step (parameter sharing)

### Variants of RNN

#### Basic/Vanilla RNN

**Structure**: Simple recurrent connection

**Problem**: Vanishing gradient issue - struggles with long sequences

**Use**: Short sequences only (rarely used in practice)

#### LSTM (Long Short-Term Memory)

**Innovation**: Introduced gates to control information flow

**Components**:

- **Forget Gate**: Decides what to remove from memory
- **Input Gate**: Decides what new information to store
- **Output Gate**: Decides what to output

**Advantage**: Can learn long-range dependencies (hundreds of steps)

**Use**: Most common RNN variant, default choice for many tasks

#### GRU (Gated Recurrent Unit)

**Innovation**: Simplified version of LSTM with fewer gates

**Components**:

- **Reset Gate**: Controls how much past information to forget
- **Update Gate**: Controls how much new information to add

**Advantage**: Faster training than LSTM, similar performance

**Use**: When computational efficiency is important

### Use Cases

**Best For**:

- ✅ Sequential data where order matters
- ✅ Time series predictions
- ✅ Natural language processing
- ✅ Speech recognition
- ✅ Video analysis
- ✅ Any temporal pattern recognition

**Examples**:

1. **Language Translation**
   - Input: Sentence in English (sequence)
   - Output: Sentence in French (sequence)
   - Application: Google Translate (though now uses Transformers)

2. **Text Generation**
   - Input: Beginning of sentence
   - Output: Next words in sequence
   - Application: Autocomplete, creative writing assistance

3. **Speech Recognition**
   - Input: Audio waveform (temporal sequence)
   - Output: Text transcription
   - Application: Voice assistants

4. **Stock Price Prediction**
   - Input: Historical price sequence
   - Output: Future price predictions
   - Application: Financial forecasting

5. **Video Action Recognition**
   - Input: Sequence of video frames
   - Output: Action classification
   - Application: Security, sports analysis

6. **Music Generation**
   - Input: Beginning notes
   - Output: Continuation of melody
   - Application: AI composition

### Advantages

- ✅ Handles sequential and temporal data
- ✅ Variable-length input/output
- ✅ Captures temporal dependencies
- ✅ Shares parameters across time steps
- ✅ Can model context and memory

### Limitations

- ❌ Sequential processing (can't parallelize easily)
- ❌ Slow training on long sequences
- ❌ Vanishing gradient problem (basic RNN)
- ❌ Struggles with very long-range dependencies
- ❌ Transformers now preferred for many NLP tasks

## Transformer Architecture

### What is Transformer?

Revolutionary architecture introduced in 2017 ("Attention Is All You Need" paper) that processes entire sequences simultaneously using self-attention mechanisms, rather than step-by-step like RNNs.

### Key Innovation: Self-Attention

**The Problem with RNNs**:

- Process sequences one step at a time (sequential)
- Information must pass through many steps
- Long-range dependencies are difficult
- Can't leverage parallel processing effectively

**Transformer Solution**:

- Process all sequence elements simultaneously (parallel)
- Each element directly attends to all other elements
- Can capture long-range dependencies immediately
- Highly parallelizable (fast training)

### Self-Attention Mechanism

**Core Idea**: Determine which parts of the input are most relevant to each other

**How It Works**:

1. **Query, Key, Value**: Each word creates three representations
2. **Attention Scores**: Calculate how much each word should attend to every other word
3. **Weighted Sum**: Combine information based on attention scores
4. **Output**: Contextual representation for each word

**Example**:

```
Sentence: "The cat sat on the mat"

When processing "sat":
- High attention to: "cat" (who sat), "mat" (where)
- Low attention to: "the" (less informative)

Result: "sat" representation enriched with relevant context
```

**Visualization**:

```
Word: "cat"
Attends to:
- "The" (15% attention)
- "cat" (25% attention) - attending to itself
- "sat" (35% attention) - strong relationship (subject-verb)
- "on" (5% attention)
- "the" (5% attention)
- "mat" (15% attention)
```

### Architecture Components

#### 1. Multi-Head Attention

**Purpose**: Learn different types of relationships simultaneously

**How It Works**:

- Run multiple attention mechanisms in parallel (e.g., 8 heads)
- Each head can learn different patterns:
  - Head 1: Subject-verb relationships
  - Head 2: Object relationships
  - Head 3: Adjective-noun relationships
  - etc.
- Combine outputs for rich representation

#### 2. Positional Encoding

**Problem**: Without sequential processing, model doesn't know word order

**Solution**: Add positional information to input embeddings

**Methods**:

- Sinusoidal functions (original paper)
- Learned positional embeddings
- Relative position encodings

#### 3. Feed-Forward Networks

**Purpose**: Process each position independently after attention

**Structure**: Two linear layers with non-linearity

**Function**: Transform the attended representations

#### 4. Layer Normalization and Residual Connections

**Purpose**: Stabilize training and enable very deep networks

**Residual Connections**: Add input directly to output (skip connections)

**Layer Normalization**: Normalize activations for stable training

### Transformer Architecture Overview

```
Input Sequence
      ↓
Embedding + Positional Encoding
      ↓
[Repeated N times (e.g., 12 layers)]
┌─────────────────────────────────┐
│ Multi-Head Self-Attention       │
│          ↓                       │
│ Add & Normalize (Residual)      │
│          ↓                       │
│ Feed-Forward Network            │
│          ↓                       │
│ Add & Normalize (Residual)      │
└─────────────────────────────────┘
      ↓
Output Layer
```

### Use Cases

**Best For**:

- ✅ Natural language processing
- ✅ Language translation
- ✅ Text generation
- ✅ Question answering
- ✅ Long-range dependencies
- ✅ Any sequence modeling task

**Examples**:

1. **Language Models** (GPT, BERT)
   - Input: Text sequence
   - Output: Predictions, embeddings, generations
   - Application: ChatGPT, language understanding

2. **Machine Translation**
   - Input: Sentence in source language
   - Output: Sentence in target language
   - Application: High-quality translation services

3. **Text Summarization**
   - Input: Long document
   - Output: Concise summary
   - Application: Document processing

4. **Question Answering**
   - Input: Context + Question
   - Output: Answer extracted or generated
   - Application: Search, assistants

5. **Vision Transformers (ViT)**
   - Input: Image patches treated as sequence
   - Output: Image classification
   - Application: Competing with CNNs for vision tasks

### Famous Transformer Models

**BERT** (2018): Bidirectional encoder, excellent for understanding tasks

**GPT** (2018-present): Autoregressive decoder, excellent for generation (ChatGPT based on this)

**T5** (2019): Text-to-text framework, unified approach to NLP tasks

**Vision Transformer (ViT)** (2020): Transformers for image classification

**DALL-E** (2021): Transformer for image generation from text

### Advantages

- ✅ Parallelizable (fast training)
- ✅ Captures long-range dependencies effectively
- ✅ State-of-the-art performance on most NLP tasks
- ✅ Can process very long sequences (with modifications)
- ✅ Transfer learning works exceptionally well
- ✅ Attention provides interpretability

### Limitations

- ❌ Quadratic memory complexity (sequence length²)
- ❌ Very computationally expensive
- ❌ Requires massive amounts of training data
- ❌ Large model sizes (billions of parameters)
- ❌ High energy consumption for training

## Architecture Comparison

### Quick Reference Table

| Architecture | Best For | Strengths | Weaknesses |
|-------------|----------|-----------|------------|
| **FFNN** | Tabular data, simple tasks | Simple, fast, interpretable | No spatial/temporal awareness |
| **CNN** | Images, spatial data | Spatial features, parameter sharing | Not for sequences, fixed input size |
| **RNN/LSTM** | Sequences, time series | Memory, variable length | Slow training, vanishing gradients |
| **Transformer** | NLP, long sequences | Parallelizable, long-range dependencies | Very expensive, huge data needed |

### Data Type Recommendations

| Data Type | Recommended Architecture | Why |
|-----------|-------------------------|-----|
| Tabular/Structured | FFNN | Independent features, no spatial/temporal structure |
| Images | CNN | Spatial relationships, local patterns |
| Text | Transformer (or RNN) | Sequential, long-range dependencies |
| Time Series | RNN/LSTM or Transformer | Temporal patterns, memory |
| Audio/Speech | RNN/LSTM or Transformer | Sequential audio, temporal patterns |
| Video | CNN + RNN or 3D CNN | Spatial (per frame) + Temporal (across frames) |

### Performance vs Cost

| Architecture | Training Speed | Inference Speed | Memory Usage | Typical Model Size |
|-------------|---------------|-----------------|--------------|-------------------|
| **FFNN** | Fast | Very Fast | Low | Small-Medium |
| **CNN** | Moderate-Fast | Fast | Moderate-High | Medium-Large |
| **RNN/LSTM** | Slow | Moderate | Moderate | Medium |
| **Transformer** | Fast (parallel) | Fast-Moderate | Very High | Very Large |

## Choosing the Right Architecture

### Decision Framework

**Step 1: Identify Your Data Type**

- Tabular/Structured → FFNN
- Images → CNN
- Text → Transformer (or RNN for smaller projects)
- Time Series → RNN/LSTM or Transformer
- Audio → RNN/LSTM or Transformer
- Video → CNN + RNN or 3D CNN

**Step 2: Consider Your Resources**

- Limited compute → FFNN or small CNN
- Moderate compute → CNN or RNN
- High compute (GPUs) → Transformer or large CNN
- Mobile/Edge deployment → Efficient architectures, compressed models

**Step 3: Evaluate Your Dataset**

- Small dataset (< 10K) → FFNN or simple architectures
- Medium dataset (10K-100K) → CNN or RNN
- Large dataset (100K+) → Deep CNN or Transformer
- Very large dataset (1M+) → Large Transformers

**Step 4: Define Success Criteria**

- Need interpretability → FFNN or simpler models
- Need state-of-the-art → Transformer or advanced CNN
- Need real-time performance → Optimized CNN or FFNN
- Cost-sensitive → Smaller, efficient models

### Practical Examples

**Scenario 1: Classifying Handwritten Digits**

- Data: 28×28 grayscale images
- Recommendation: CNN (e.g., LeNet)
- Why: Images, spatial patterns, proven solution

**Scenario 2: Predicting Customer Churn**

- Data: Customer demographics and usage (tabular)
- Recommendation: FFNN or XGBoost
- Why: Structured data, no spatial/temporal patterns

**Scenario 3: Building a Chatbot**

- Data: Conversational text
- Recommendation: Transformer (fine-tuned BERT or GPT)
- Why: Natural language, context understanding, long conversations

**Scenario 4: Stock Price Forecasting**

- Data: Historical prices and indicators (time series)
- Recommendation: LSTM or Transformer
- Why: Temporal patterns, need for memory

**Scenario 5: Autonomous Vehicle Vision**

- Data: Real-time camera feeds (video)
- Recommendation: CNN for object detection + temporal processing
- Why: Real-time spatial understanding, object recognition

## Key Takeaways

1. **Different architectures for different data types** - match architecture to problem
2. **FFNNs are simple and effective** for tabular data
3. **CNNs revolutionized computer vision** through spatial feature learning
4. **RNNs handle sequential data** with memory mechanisms
5. **Transformers dominate modern NLP** and are expanding to other domains
6. **No one-size-fits-all solution** - choose based on data, resources, and requirements
7. **Start simple, then scale** - don't jump to Transformers when FFNN might suffice

## Next Steps

- Learn about [Large Language Models](04-Large-Language-Models.md) - Transformers for language
- Explore [Generative AI](05-Generative-AI.md) - creating content with neural networks
- Understand [ML Tools and Frameworks](08-ML-Tools-and-Frameworks.md) - implementing these architectures

---

[← Back to Deep Learning](02-Deep-Learning.md) | [Next: Large Language Models →](04-Large-Language-Models.md)
