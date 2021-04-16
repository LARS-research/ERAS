# ERAS
The code for our paper ["Efficient Relation-aware Scoring Function Search for Knowledge Graph Embedding"], which has been accepted by ICDE 2021.

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper

    @inproceedings{di2021eras,
      title={Efficient Relation-aware Scoring Function Search for Knowledge Graph Embedding},
      author={Shimin DI, Quanming YAO, Yongqi ZHANG, and Lei CHEN},
      booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)},
      pages={},
      year={20221},
      organization={IEEE}
    }

## Instructions
For the sake of ease, a quick instruction is given for readers to reproduce the whole process.
Note that the programs are tested on Linux(Ubuntu release 16.04), Python 3.7 from Anaconda 4.5.11.

Install PyTorch (>0.4.0)
    
    conda install pytorch -c pytorch

Search and train the searched scoring functions from scratch
    python /one-shot-search/evaluate.py


Related AutoML papers (ML Research group in 4Paradigm)
- Searching to Sparsify Tensor Decomposition for N-ary Relational Data. Webconf 2021 [paper](), [code](https://github.com/AutoML-4Paradigm/S2S)
- Interstellar: Searching Recurrent Architecture for Knowledge Graph Embedding. NeurIPS 2020 [paper](https://arxiv.org/pdf/1911.07132.pdf), [code](https://github.com/AutoML-4Paradigm/Interstellar)
- AutoSF: Searching Scoring Functions for Knowledge Graph Embedding. ICDE 2020 [paper](https://arxiv.org/pdf/1904.11682.pdf), [code](https://github.com/AutoML-4Paradigm/AutoSF)
- Simple and Automated Negative Sampling for Knowledge Graph Embedding. ICDE 2019 [paper](https://arxiv.org/abs/1812.06410), [code](https://github.com/yzhangee/NSCaching)

