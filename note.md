
```bash

conda create -n kotomamba python=3.11
conda activate kotomamba

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y


cd kotomamba
pip install packaging wheel
pip install causal-conv1d>=1.1.0
pip install -e .

cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .


conda install mpi4py

```

```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/monology/pile-uncopyrighted


```

```bash
mpirun -np 1 \
    --npernode 1 \
    python pretrain.py \
    --tokenizer_name EleutherAI/gpt-neox-20b \
    --model_name state-spaces/mamba-130m \
    --from_scratch \
    --enable_fsdp


python -m torch.distributed.launch \
    pretrain.py \
    --tokenizer_name EleutherAI/gpt-neox-20b \
    --model_name state-spaces/mamba-130m \
    --from_scratch \
    --dataset alpaca_dataset \
    --enable_fsdp



```
