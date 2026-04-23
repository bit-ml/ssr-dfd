# BRAVEn VSR feature extraction

- First extract mouth regions and audio using AV-HuBERT preprocessing code (see `avhubert` folder).
- `git clone https://github.com/ahaliassos/raven.git`.
- Download the BRAVEn VSR high-resource Large w/ ST model (link [here](https://drive.google.com/file/d/1bU-bFxEiXNXNoOLKaAz6_9j7XuE7f_6r/view?usp=sharing)) and place it `raven/ckpts/`.
- Replace the files found in this folder in their corresponding place inside the `raven` folder.
```
cd raven_modified
cp finetune_learner.py test.py my_environment.yml ../raven/.
cp data/* ../raven/data/.
cp -r conf/* ../raven/conf/.
cd ../raven
```
- Change the following files:
    - `conf/config_test.yaml` -> `output_dir` field; where to save the features
    - `conf/data/dataset/lrs3.yaml` -> `test_csv` and `root_path`; where the preprocessed videos and the corresponding CSV file are located.
- Set the conda env (`conda env create -f my_environment.yml`). You can also try using the original `environment.yml` but it might fail due to some package errors; results should not differ.

Finally, run the following command (which can be also found in `raven/scrips/testing/vsr/lrse3/large_lrs3vox2avs_self_braven.sh`):

```
python raven/test.py \
    data.modality=video \
    data/dataset=lrs3 \
    experiment_name=vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven_test \
    model/visual_backbone=resnet_transformer_large \
    model.pretrained_model_path=ckpts/vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven.pth \
```