import warnings

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning, module="torch")    
from src.main import load_langgraph_agenticai_app
if __name__ == "__main__":
   load_langgraph_agenticai_app()     