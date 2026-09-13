## Train 

```bash
bash adzoo/drivetransformer/dist_train.sh adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py 8 #N_GPUS
```

## Open Loop Eval

```bash
bash adzoo/drivetransformer/dist_test.sh adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py ckpts/drivetransformer_large.pth 1 #N_GPUS
```

## Closed Loop Eval    

- **STEP 1: Clone Bench2Drive repo**

```bash
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
```
And make sure you have installed CARLA following the step 8 in [doc](./INSTALL.md).

- **STEP 2: Link this repo to Bench2Drive**

```bash
cd PATH_TO_Bench2Drive
ln -s DriveTransformer  ./   
mkdir team_code
ln -s DriveTransformer/team_code/* ./team_code    

```
- **STEP 3: Run evaluation**

```bash
cd PATH_TO_Bench2Drive
cp DriveTransformer/run_evaluation_multi_dtr.sh leaderboard/scripts
bash leaderboard/scripts/run_evaluation_multi_dtr.sh
```

You can find more details about Bench2Drive evaluation [here](https://github.com/Thinklab-SJTU/Bench2Drive?tab=readme-ov-file#eval-tools).
