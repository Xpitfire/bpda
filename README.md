# Blancing Principle for Domain Adaptation

![alt text](https://github.com/Xpitfire/bpda/blob/master/figures/balancing_principle.png "Logo Title Text 1")

## Abstract

We address the unsolved algorithm design problem of choosing a justified regularization parameter in unsupervised domain adaptation. This problem is intriguing as no labels are available in the target domain. Our approach starts with the observation that the widely-used approach of minimizing the source error, weighted by a distance measure between source and target feature representations, shares characteristics with regularized ill-posed inverse problems.
Regularization parameters in inverse problems can be chosen by the fundamental principle of balancing approximation and sampling errors. We use this principle to balance learning errors and domain distance in a target error bound. As a result, we obtain a theoretically justified rule for the choice of the regularization parameter. In contrast to the state of the art, our approach allows source and target distributions with disjoint supports. An empirical comparative study on benchmark datasets underpins the performance of our approach.

## Installing

1. Clone repository
```bash
git clone https://github.com/Xpitfire/bpda
cd bpda
```

2. Create a python 3 conda environment
```bash
conda env create -f environment.yml
```

3. Install package
```bash
pip install -e .
```

4. Ensure that all required temp directories are available

  * `tmp`
  * `runs`
  * `data`

## Compute Results

1. Train domain adaptation method with balancing principle by calling the `bp` configs:
```bash
CUDA_VISIBLE_DEVICES=<device-id> PYTHONPATH=. python scripts/train.py --config configs/<your-bp-config>.json
```
```bash
# running a CMD experiment with BP
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train.py --config configs/config.minidomainnet_bp_cmd.json.json
# running a MMD experiment with BP
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train.py --config configs/config.minidomainnet_bp_mmd.json.json
```

2. Evaluate results
Set the respective `base_dir` and `method` setting in the `viz/results_extractor_MiniDomainNet.py` file and run:
```bash
PYTHONPATH=. python viz/results_extractor_MiniDomainNet.py
```

## References

* [Moment Matching for Multi-Source Domain Adaptation](http://ai.bu.edu/M3SDA/)
* [Amazon product data](https://jmcauley.ucsd.edu/data/amazon/)
* [Unsupervised Domain Adaptation by Backpropagation](https://github.com/fungtion/DANN)
* [Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation](https://github.com/thuml/Deep-Embedded-Validation)


```latex
@article{Zellinger:21,
  title={The balancing principle for parameter choice in distance-regularized domain adaptation},
  author={Werner Zellinger and Natalia Shepeleva and Marius-Constantin Dinu and Hamid Eghbal-zadeh and Ho\'an Nguyen Duc and Bernhard Nessler and Sergei V.~Pereverzyev and Bernhard A. Moser},
  journal={NeurIPS},
  year={2021}
}
```
