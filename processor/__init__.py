"""
红外小目标检测序列后处理器包

包含三种不同的序列后处理器：
1. SimpleSequenceProcessor - 简单处理器
2. ImprovedSequenceProcessor - 改进处理器  
3. BalancedSequenceProcessor - 平衡处理器
"""

from .simple_sequence_processor import SimpleSequenceProcessor
from .improved_sequence_processor import ImprovedSequenceProcessor
from .balanced_sequence_processor import BalancedSequenceProcessor

__all__ = [
    'SimpleSequenceProcessor',
    'ImprovedSequenceProcessor', 
    'BalancedSequenceProcessor'
]

__version__ = '1.0.0'
__author__ = 'Your Name'
__description__ = '红外小目标检测序列后处理器包' 