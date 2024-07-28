from slot_attention.params import SlotAttentionParams
from slot_attention.train import main
from slot_attention.inference import run_inference
import argparse

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run Slot Attention."
    )
    parser.add_argument(
        "--mode", default="train", help="available options: train/test"
    )
    args = parser.parse_args()
    
    if args.mode == "train":
        params = SlotAttentionParams(max_epochs=150)
        main(params)
    elif args.mode == "test":
        params = SlotAttentionParams()
        run_inference(params)
    else:
        print("unknown mode")
    
