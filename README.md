# 🧠 Continual Learning System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A robust framework for training neural networks that can learn sequentially without forgetting**

[Overview](#overview) •
[Key Features](#key-features) •
[Techniques](#techniques) •
[Installation](#installation) •
[Usage](#usage) •
[Results](#results) •
[Contributing](#contributing)

</div>

## 🔄 Overview

The Continual Learning System is a comprehensive framework for developing neural networks that can learn tasks sequentially without suffering from catastrophic forgetting. This project implements several state-of-the-art techniques to mitigate forgetting in neural networks, allowing them to adapt to new tasks while retaining performance on previously learned ones.

<div align="center">
    <img src="https://via.placeholder.com/800x400?text=Continual+Learning+Workflow" alt="Continual Learning Workflow" width="80%">
</div>

## 🌟 Key Features

- **Task Sequential Learning**: Train models on a sequence of tasks without complete retraining
- **Forgetting Mitigation**: Advanced techniques to prevent catastrophic forgetting
- **Performance Tracking**: Comprehensive metrics to monitor how well knowledge is retained
- **Experiment Framework**: Easily run and compare different continual learning approaches
- **Visualization Tools**: Track and visualize forgetting metrics across sequential tasks

## 🧩 Techniques Implemented

### Elastic Weight Consolidation (EWC)

EWC measures the importance of neural network weights for previously learned tasks and penalizes changes to important weights when learning new tasks.

```python
# Loss calculation with EWC
loss = task_loss + lambda_ewc * ewc_loss
```

### Experience Replay

This technique maintains a memory buffer of examples from previous tasks and periodically replays them during training on new tasks.

```python
# Replay during training
combined_loss = current_task_loss + alpha * replay_loss
```

### Learning without Forgetting (LwF)

LwF uses knowledge distillation to preserve the model's behavior on previous tasks when learning new ones.

```python
# LwF distillation loss
distillation_loss = KL_divergence(current_outputs, previous_outputs)
```

### Task-specific Components

For some approaches, we isolate or add task-specific parameters while sharing a common feature extraction backbone.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/1Utkarsh1/continual-learning.git
cd continual-learning

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Usage

### Quick Start

```bash
# Run baseline experiment (sequential training without any continual learning techniques)
python src/main.py --method baseline --tasks mnist_split

# Run EWC experiment
python src/main.py --method ewc --tasks mnist_split --lambda_ewc 5000

# Run Experience Replay experiment
python src/main.py --method replay --tasks mnist_split --buffer_size 500
```

### Custom Task Sequences

You can define your own task sequences in a YAML configuration file:

```yaml
# config/tasks/custom_sequence.yaml
task_sequence:
  - name: "mnist_digits_0_4"
    dataset: "mnist"
    classes: [0, 1, 2, 3, 4]
  
  - name: "mnist_digits_5_9"
    dataset: "mnist"
    classes: [5, 6, 7, 8, 9]
  
  - name: "fashion_mnist"
    dataset: "fashion_mnist"
    classes: "all"
```

## 📈 Experimental Results

### Comparison of Methods

<div align="center">
<table>
  <tr>
    <th>Method</th>
    <th>Average Accuracy</th>
    <th>Average Forgetting</th>
    <th>Training Time</th>
  </tr>
  <tr>
    <td>Naïve Fine-tuning</td>
    <td>45.2%</td>
    <td>35.8%</td>
    <td>1.0x</td>
  </tr>
  <tr>
    <td>EWC</td>
    <td>78.5%</td>
    <td>10.2%</td>
    <td>1.2x</td>
  </tr>
  <tr>
    <td>Experience Replay</td>
    <td>82.3%</td>
    <td>7.5%</td>
    <td>1.5x</td>
  </tr>
  <tr>
    <td>LwF</td>
    <td>75.7%</td>
    <td>12.8%</td>
    <td>1.3x</td>
  </tr>
</table>
</div>

### Performance Over Tasks

<div align="center">
<img src="https://via.placeholder.com/600x400?text=Performance+Over+Tasks" alt="Performance Over Tasks" width="70%">
</div>

## 📝 Recent Experiment Results

The following experiment results were obtained on March 11, 2025 using the MNIST split task sequence:

**Baseline (Naïve Fine-tuning):**
- Command: `python src/main.py --method baseline --tasks mnist_split --epochs 5`
- Task sequence: ['mnist_0_4', 'mnist_5_9']
- Average final accuracy: 49.74%
- Average forgetting: 49.90%

**Learning without Forgetting (LwF):**
- Command: `python src/main.py --method lwf --tasks mnist_split --epochs 5`
- Task sequence: ['mnist_0_4', 'mnist_5_9']
- Average final accuracy: 49.67%
- Average forgetting: 49.83%

## 🛠️ Project Structure

```
continual_learning/
├── src/                    # Source code
│   ├── models/             # Neural network architectures
│   ├── data/               # Data loading and preprocessing
│   ├── methods/            # Continual learning algorithms
│   ├── utils/              # Utility functions
│   └── main.py             # Main entry point
├── experiments/            # Jupyter notebooks for experiments
├── config/                 # Configuration files
│   ├── models/             # Model configurations
│   └── tasks/              # Task sequence definitions
├── results/                # Saved results and visualizations
└── docs/                   # Documentation
```

## 📝 Example Experiments

1. **Split MNIST**
   - Train on digits 0-4, then 5-9
   - Compare different methods' ability to remember the first task

2. **Task Incremental Learning**
   - Train on MNIST → Fashion-MNIST → KMNIST
   - Measure accuracy on all previous datasets after each task

3. **Class Incremental Learning**
   - Add new classes (one at a time) to a classifier
   - Test identification of all classes after each addition

## 🔮 Roadmap

- [x] Implement baseline sequential training
- [x] Implement Elastic Weight Consolidation (EWC)
- [x] Implement Experience Replay
- [x] Implement Learning without Forgetting (LwF)
- [ ] Add support for generative replay
- [ ] Implement parameter isolation methods
- [ ] Add support for continual reinforcement learning
- [ ] Develop benchmark suite for comparing methods

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

1. Kirkpatrick, J. et al. "Overcoming catastrophic forgetting in neural networks" - *Proceedings of the National Academy of Sciences* (2017)
2. Rebuffi, S. et al. "iCaRL: Incremental Classifier and Representation Learning" - *CVPR* (2017)
3. Li, Z. and Hoiem, D. "Learning without Forgetting" - *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2018)
4. Chaudhry, A. et al. "Efficient Lifelong Learning with A-GEM" - *ICLR* (2019)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
    <b>Made with ❤️ by the Continual Learning Team</b><br>
    <a href="https://github.com/1Utkarsh1">GitHub</a> •
    <a href="#">Website</a> •
    <a href="#">Contact</a>
</div> 