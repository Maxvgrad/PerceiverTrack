# Installation

1. Clone and enter this repository:
    ```
    git clone git@github.com:timmeinhardt/trackformer.git
    cd trackformer
    ```

2. Install packages for Python 3.8:

    1. `conda create -n perceiver_track python=3.10`
    2. `conda activate perceiver_track`
    3. `conda install conda-forge::motmetrics=1.2.0`
    3. `pip install sacred==0.8.5` it didn't work with conda `conda install conda-forge::sacred=0.8.5`
    3. `pip install visdom==0.1.8.9` it didn't work with conda `conda install conda-forge::visdom=0.1.8.9`
    3. `conda install conda-forge::pycocotools`
    3. `conda install conda-forge::einops`
    3. `conda install scikit-image==0.17.2` # it is needed for dataset conversion
    3. `conda install conda-forge::tqdm=4.46.1`
    3. `conda install conda-forge::seaborn=0.10.1`
    3. `conda install conda-forge::opencv`
    3. `pip install perceiver-pytorch`
    3. `conda install conda-forge::wandb`
    3. `pip install nuscenes-devkit` # nuimages dataset 

    5. Install MultiScaleDeformableAttention
       package: `python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install`

3. Download and unpack datasets in the `data` directory:

    1. [MOT17](https://motchallenge.net/data/MOT17/):

        ```
        wget https://motchallenge.net/data/MOT17.zip
        unzip MOT17.zip
        python src/generate_coco_from_mot.py
        ```

    2. (Optional) [MOT20](https://motchallenge.net/data/MOT20/):

        ```
        wget https://motchallenge.net/data/MOT20.zip
        unzip MOT20.zip
        python src/generate_coco_from_mot.py --mot20
        ```

    3. (Optional) [MOTS20](https://motchallenge.net/data/MOTS/):

        ```
        wget https://motchallenge.net/data/MOTS.zip
        unzip MOTS.zip
        python src/generate_coco_from_mot.py --mots
        ```

    4. (Optional) [CrowdHuman](https://www.crowdhuman.org/download.html):

        1. Create a `CrowdHuman` and `CrowdHuman/annotations` directory.
        2. Download and extract the `train` and `val` datasets including their corresponding `*.odgt` annotation file into the `CrowdHuman` directory.
        3. Create a `CrowdHuman/train_val` directory and merge or symlink the `train` and `val` image folders.
        4. Run `python src/generate_coco_from_crowdhuman.py`
        5. The final folder structure should resemble this:
            ~~~
            |-- data
                |-- CrowdHuman
                |   |-- train
                |   |   |-- *.jpg
                |   |-- val
                |   |   |-- *.jpg
                |   |-- train_val
                |   |   |-- *.jpg
                |   |-- annotations
                |   |   |-- annotation_train.odgt
                |   |   |-- annotation_val.odgt
                |   |   |-- train_val.json
            ~~~
           
    4. [nuimages](https://www.nuscenes.org/nuimages):
       1. Download dataset extract it in `./data/nuimages`
       2. Run (optinally you can specify which cameras use images)

        ```
       python src/generate_coco_from_nuimages.py \
         --data-root ./data/nuimages \
         --version v1.0-mini  v1.0-test  v1.0-train  v1.0-val \
         --out-dir ./data/nuimages/annotations/ \
         --cameras CAM_FRONT \
         --extra-tag cam-front_nuimages \
         --categories pedestrian
       ```

3. Download and unpack pretrained TrackFormer model files in the `models` directory:

    ```
    wget https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip
    unzip trackformer_models_v1.zip
    ```

4. (optional) The evaluation of MOTS20 metrics requires two steps:
    1. Run Trackformer with `src/track.py` and output prediction files
    2. Download the official MOTChallenge [devkit](https://github.com/dendorferpatrick/MOTChallengeEvalKit) and run the MOTS evaluation on the prediction files

In order to configure, log and reproduce our computational experiments, we structure our code with the [Sacred](http://sacred.readthedocs.io/en/latest/index.html) framework. For a detailed explanation of the Sacred interface please read its documentation.
