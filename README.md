[CVPR 2026] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers.

---

## 🚀 Environment Setup

Since our mechanisms are built upon established frameworks, please follow the original setup instructions:
- **Image Classification:** Refer to the official [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) repository for dependency and dataset setup.
- **Cross-modal Retrieval:** Refer to the official [SA-AET](https://github.com/jiaxiaojunQAQ/SA-AET) repository for multi-modal environment setup (e.g., MS-COCO configuration, Transformers).
---

## 🛠️ Usage

### TransferAttack (Image Classification)


**Running an Attack Example:**
Assuming you are in the `TransferAttack` root directory with properly configured datasets, run:

```bash
# Generate adversarial examples using MDCS-OPS
python main.py --input_dir ./data --output_dir adv_data/mdcsmi/resnet50 --attack mdcsmi --model resnet50

# Evaluate the generated adversarial examples
python main.py --input_dir ./data --output_dir adv_data/mdcsmi/resnet50 --eval
```

### Retrieval (Cross-modal Retrieval)

The `Retrieval` folder contains the `plus` implementations which denote our integration of the MDCS step-size adaptive mechanism into existing baselines:
- `SA_AET_plus.py`
- `SGAttacker_plus.py`
- `DRAttacker_plus.py`

**Running an Attack Example:**
Assuming you are in the `Retrieval` root directory with properly configured datasets, run:

```bash
# Evaluate the generated adversarial examples on SGAttacker + MDCS
python eval.py --config ./configs/Retrieval_flickr.yaml --attack mdcssga

```

---

## 🙏 Acknowledgements
We would like to thank the authors of the following outstanding open-source repositories, which significantly facilitated our research and codebase construction:
- [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)
- [SA-AET](https://github.com/jiaxiaojunQAQ/SA-AET)
- [VLPTransferAttack](https://github.com/SensenGao/VLPTransferAttack)

