"""Configuration and argument parsing for experiments."""
import argparse
import os
from datetime import datetime

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Experiment Set Up",
        description="Test randomization ideas on soft trees.",
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Select mode to run")

    # Torch subparser
    torch_parser = subparsers.add_parser("torch", help="Run with torch framework")
    torch_parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    torch_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    torch_parser.add_argument("--num_trees", type=int, required=True, help="Number of trees")
    torch_parser.add_argument("--max_depth", type=int, required=True, help="Max depth for trees")
    torch_parser.add_argument("--combine_output", action="store_true", default=True, 
                             help="Combine output into leaf_dims")
    torch_parser.add_argument("--subset_selection", action="store_true", 
                             help="Use Hadamard product for random feature selection")
    torch_parser.add_argument("--subset_share", type=float, 
                             help="Share of features to use for subset selection")
    torch_parser.add_argument("--epochs", type=int, required=True, help="Training epochs")
    torch_parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    torch_parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    torch_parser.add_argument("--bootstrap", action="store_true", help="Whether to bootstrap")
    torch_parser.add_argument("--noise_level", type=float, default=0.15,
                             help="Proportion of labels to shuffle (0-1)")
    
    # Add other subparsers (sklearn, boost, tf) here...
    
    return parser.parse_args()

def setup_logging(args):
    """Set up logging for the experiment."""
    current_time = datetime.now().strftime("%m-%d-%H-%M")
    log_dir = "outputs/single_run"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir,
        f"run_logs_{args.dataset_name}_subsetselection_{args.subset_selection}_{current_time}.log"
    )
    
    import logging
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    
    return log_file