"""BasicSR discovery entry for BiSTAVSR.

The implementation remains in ``BiSTAVSR.py`` so existing direct imports keep
working, while this ``*_arch.py`` module allows BasicSR's automatic registry
scanner to discover the network from YAML configurations.
"""

from .BiSTAVSR import BiSTAVSR

__all__ = ['BiSTAVSR']
