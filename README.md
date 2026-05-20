# SSR-DFD

[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2511.17181)

**Official PyTorch Implementation of the Paper:**

> **Dragoș-Alexandru Boldisor, Ștefan Smeu, Dan Oneață and Elisabeta Oneață**  
> [Investigating self-supervised representations for audio-visual deepfake detection](https://arxiv.org/abs/2511.17181)  
> *CVPR, 2026*


## Features
To extract the best set of features for each modality reported in the paper (Wav2Vec2, BRAVEn Video and AV-HuBERT) see the `feature_extraction` folder.

## Linear probing
Code and checkpoints for linear probing can be found in the `linear_probing` folder.

### Installation
```bash
conda create -n ssr_dfd python=3.10
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Train
To train the models, consult the `configs/train_config.yaml` config file and fill the necessary fields. After that, run the following command:
```bash
python train_test.py --config configs/train_config.yaml
```

The csv required need to have two columns: `path` containing the relative path from the `root_dir` specified in the config, and `label` containing `0` (for real) or `1` (for fake). The `input_type` fields (both from `data_info` and from `model_hparams`) need to have the SAME VALUE!

The checkpoints, logs (in the form of csv) and hyperparameter used (set in config file) will be saved in the specified output folder.

### Test
To test a trained model, you need to fill the `configs/test_config.yaml` config file (the fields have similar meaning to the ones used in `train_config.yaml`).

## License

<p xmlns:cc="http://creativecommons.org/ns#">The code is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0 <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

## Citation

If you find this work useful in your research, please cite it.

```
@InProceedings{ssr-dfd,
  title={Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection},
  author={Boldisor, Dragos-Alexandru and Smeu, Stefan and Oneata, Dan and Oneata, Elisabeta},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```