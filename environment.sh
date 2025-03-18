conda create -n corgs python=3.8.1
conda activate corgs
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pip==22.3.1 -c conda-forge
pip install submodules/diff-gaussian-rasterization-confidence/
pip install submodules/simple-knn/
conda install -c conda-forge plyfile==0.8.1 tqdm