# Script to run all of the zinc experiments
#todo - consider pulling the github commit id and whether there are any diffs in important files and reporting this via wandb

python main_zinc.py --config 'configs/GatedGCN_ZINC_NoPE.json' --truncate_to=20 --batch_size=5 --wandb_run_name="proto_zinc_trunc_20_bs_5_script" --wandb
