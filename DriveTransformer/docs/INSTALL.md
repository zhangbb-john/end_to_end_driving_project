## Follow these steps to install the environment
- **STEP 1: Create environment**
    ```
    ## python3.8 should be strictly followed.
    conda create -n drivetransformer python=3.8
    conda activate drivetransformer
    ```
- **STEP 2: Install cudatoolkit**
    ```
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```
- **STEP 3: Install torch and xformers**
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
    ```
- **STEP 4: Set environment variables**
    ```
    # cuda 11.8 and GCC 9.4 is strongly recommended. Otherwise, it might encounter errors.
    export PATH=YOUR_GCC_PATH/bin:$PATH
    export CUDA_HOME=YOUR_CUDA_PATH/
    ```
- **STEP 5: Install ninja and packaging**
    ```
    pip install ninja packaging
    ```
- **STEP 6: Install our repo**
    ```
    pip install -v -e .
    ```

- **STEP 7: Download pretrained weights.**
    create directory `ckpts`

    ```
    mkdir ckpts 
    ```
    Download `resnet50-19c8e357.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth) or [Baidu Cloud](https://pan.baidu.com/s/1LlSrbYvghnv3lOlX1uLU5g?pwd=1234 ) or from Pytorch official website.
  
    Download `drivetransformer_large.pth` from (**TODO**)


- **STEP 8: Install CARLA for closed-loop evaluation.**

    ```
    ## Ignore the line about downloading and extracting CARLA if you have already done so.
    ## You do not need CARLA for training and open-loop evaluation. Thus, it could be skipped until you want to run with CARLA.
    mkdir carla
    cd carla
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    tar -xvf CARLA_0.9.15.tar.gz
    cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
    cd .. && bash ImportAssets.sh
    export CARLA_ROOT=YOUR_CARLA_PATH

    ## Important!!! Otherwise, the python environment can not find carla package
    echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.8/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME

    ```
