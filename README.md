# **M**asked __**U**__-Net-Based **C**ycle-Consistent **A**dversarial **N**etworks (MUCAN) - PyTorch
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  


## 🚀 What is MUCAN?

**MUCAN** is a deep learning framework designed to make **intracortical brain-computer interfaces (iBCIs)** more stable, reliable, and adaptable—without the need for daily recalibration.

In real-world BCI systems, decoding motor intentions from neural signals becomes challenging due to biological changes and electrode variability over time. MUCAN tackles this by aligning the distribution of neural signals across sessions using a novel masked U-Net architecture design.

---

## 🧠 Why does this matter?

Traditional decoders require **frequent supervised recalibration**, which is time-consuming and not user-friendly. MUCAN enables:

- **Unsupervised domain adaptation** across days/sessions
- Improved decoder robustness to neural signal drift
- More efficient training


---

## 🧪 Quick Demo (from `tutorial.ipynb`)

### 🔹 Steps demonstrated in the tutorial:

1. **Load a sample dataset**
   - Neural recordings from the `Jango_ISO_2015` set are used.
   - Each `.npz` file contains neural firing rates and kinematics.

2. **Training processes**
   - Use `WienerFilter` (a ridge regression baseline) for initial decoding.
   - Apply the `Aligner` model (MUCAN's core) to perform alignment between source and target sessions.

3. **Visualize decoding performance**
   - Evaluate R² score after alignment by MUCAN.


