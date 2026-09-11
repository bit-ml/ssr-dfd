# TalkNet-ASD extraction

For Deepfake-Eval-2024 we extract the active-speaker segments using the official implementaion of TalkNet-ASD.

- First, clone the official implementation using the following command:
```
git clone https://github.com/TaoRuijie/TalkNet-ASD.git
```
- Then, copy the provided modified `demoTalkNet.py` file, the `get_segments.py` python script and the `run_TalkNet_deepfake_eval.sh` bash script.
```
cp demoTalkNet.py TalkNet-ASD/.
cp get_segments.py TalkNet-ASD/.
cp run_TalkNet_deepfake_eval.sh TalkNet-ASD/.
cd TalkNet-ASD/
```
- Install the required dependencies as explained in the original repo.
- Modify the `videoFolder` and `savePath` in `run_TalkNet_deepfake_eval.sh` to your desired locations.
- Run the `run_TalkNet_deepfake_eval.sh` script.
- Add the paths in `get_segments.py` and run it.