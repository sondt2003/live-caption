#!/usr/bin/env python3
"""
Script to suppress TensorFlow and other library warnings
Add this at the very beginning of inference.py
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("✅ Warning suppression enabled")
