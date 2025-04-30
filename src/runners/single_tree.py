"""Single tree experiment runner."""
import time
import logging
import torch
from softensemble import SoftTreeEnsemble
from engine import train_model
from torch.utils.data import DataLoader

def run_single_tree_experiment(train_dataset, test_dataset, input_dims, num_classes, args):
    """Run experiment with a single tree."""
    logging.info(f"Running single tree experiment")
    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Input dimensions: {input_dims}, Number of classes: {num_classes}")
    logging.info(f"Batch size: {args.batch_size}, Max depth: {args.max_depth}")
    logging.info(f"Subset selection: {args.subset_selection}, Subset share: {args.subset_share}")
    
    # Create DataLoaders for batching
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = SoftTreeEnsemble(
        num_trees=1,
        max_depth=args.max_depth,
        leaf_dims=num_classes,
        input_dim=input_dims,
        combine_output=args.combine_output,
        subset_selection=args.subset_selection,
        subset_share=args.subset_share
    )
    
    logging.info("Created Soft Tree Model")
    
    # Train model
    start_time = time.time()
    test_loss, test_accuracy, test_auc = train_model(
        model=model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )
    end_time = time.time()
    
    execution_time = (end_time - start_time) / 60
    logging.info(f"Training completed in {execution_time:.2f} minutes")
    
    # Return results
    return {
        'model': model,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'execution_time': execution_time,
        'loss': 1.0 - test_accuracy  # For hyperopt (minimize 1-accuracy)
    }