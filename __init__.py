"""ComfyUI loads this file when the repository is cloned into ComfyUI/custom_nodes (or added with ComfyUI-Manager's
"Install via Git URL"): it serves the CueSync nodes from comfyui-cuesync/, which use the cuesync library next to
them in cuesync/src and the effects in effects/."""

import importlib.util
import sys
from pathlib import Path

_pack = Path(__file__).resolve().parent / "comfyui-cuesync"
_spec = importlib.util.spec_from_file_location(f"{__name__}.comfyui_cuesync", _pack / "__init__.py",
                                               submodule_search_locations=[str(_pack)])
_nodes = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _nodes
_spec.loader.exec_module(_nodes)

NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
