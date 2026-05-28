import argparse
import json
import os
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer

def load_item_text(path:str):
    df = pd.read_csv(path, usecols=['title', 'main_category','description','details'])