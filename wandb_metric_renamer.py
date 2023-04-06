import wandb
from typing import List, Tuple, Dict

api = wandb.Api()


def rename_columns(run, changes: Dict[str, str]):
  for before in changes:
    after = changes[before]
    run.summary[after] = run.summary[before]
    del run.summary[before]
  run.update()


def do_for(paths, changes: Dict[str, str]):
  for path in paths:
    run = api.run(path)
    rename_columns(run, changes)
    run.update()


'''Example usage
ids = [
  'afnqkwfc',
  'z0uehfm2',
  '8faj94l4',
  'lgxyymh6',
  'nugl722y',
  'y4b2vhbe',
  'u3zvtvow',
  'mwqmr6nv',
  'nweky0ro',
  'kikold6e',
  '54t5ntwt',
  'ewbp53mj',
  'vqqhligs',
  '99zyrlrj',
  'xgybtyvm',
  'chq2blys',
  '4zniplkx',
  'aaxwrkq3',
  'kaqam8ke',
  'vvefk1n4',
  'm88sbfm7',
  'vu5zjjar',
  'w0f9o9j4',
    'tz2f3aga']
paths = ["marcushandley/Part II/" + id for id in ids]
changes = {
  "final_test loss": "final test loss",
    "final_val loss": "final val loss"}
do_for(paths, changes)
'''
