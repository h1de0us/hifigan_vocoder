## HiFiGAN implementation
## Installation guide

To install all the dependencies, run

```shell
cd hifigan-vocoder
pip install -r ./requirements.txt
```
To train a model, run
```
python3 train.py -c src/configs/default.json
```
To download a checkpoint, run
```
python3 upload_model.py
```
To get audios for test texts, run
```
python3 test.py -r checkpoint.pth
```
The generated audios will appear in the folder "results".


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.