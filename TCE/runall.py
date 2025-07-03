import sys
import os

# Add the tce_modules directory to the Python path
module_path = os.path.abspath(os.path.join(".", "tce_modules"))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import and run
from pretrain import main as pretrain_main
from finetune import main as finetune_main

def run_all():
    print("Starting pretraining phase...")
   # pretrain_main()
    print("Pretraining completed.\n")

    print("Starting fine-tuning phase...")
    finetune_main()
    print("Fine-tuning completed.\n")

if __name__ == "__main__":
    run_all()
