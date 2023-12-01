import torch
import os.path as osp

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
FIG_DIR = osp.join(ROOT_DIR, 'figures')
DATA_DIR = osp.join(ROOT_DIR, 'data')
RESULT_DIR = osp.join(ROOT_DIR, 'results')

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MAP_LOC = "cuda:0" if USE_CUDA else torch.device("cpu")
