"""Ensemble/bootstrap experiment runner."""
import time
import logging
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
from softensemble import SoftTreeEnsemble
from engine import train_model

def run_ensemble_experiment(train_dataset, test_dataset, input_dims, num_classes, args):
    """Run experiment with multiple trees using bootstrap."""
    logging.info(f"Running ensemble experiment with {args.num_trees} trees")
    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Input dimensions: {input_dims}, Number of classes: {num_classes}")
    logging.info(f"Batch size: {args.batch_size}, Max depth: {args.max_depth}")
    logging.info(f"Subset selection: {args.subset_selection}, Subset share: {args.subset_share}")
    
    all_preds = []
    start_time = time.time()
    
    for tree_idx in range(args.num_trees):
        logging.info(f"Training tree {tree_idx + 1}/{args.num_trees}")
        
        # Create DataLoaders (with bootstrap sampling if enabled)
        train_dataloader, test_dataloader = get_dataloaders(
            train_dataset, 
            test_dataset, 
            args.batch_size, 
            bootstrap=args.bootstrap
        )
        
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
        
        # Train model
        train_model(
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device
        )
        
        # Get predictions
        tree_preds = get_model_predictions(model, test_dataloader, args.device)
        all_preds.append(tree_preds)
    
    # Process ensemble predictions
    ensemble_results = evaluate_ensemble(
        all_preds, 
        test_dataloader, 
        num_classes
    )
    
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    logging.info(f"Ensemble training completed in {execution_time:.2f} minutes")
    logging.info(f"Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
    logging.info(f"Ensemble AUC: {ensemble_results['auc']:.4f}")
    logging.info(f"Baseline Accuracy: {ensemble_results['baseline']:.4f}")
    
    return {
        'accuracy': ensemble_results['accuracy'],
        'auc': ensemble_results['auc'],
        'baseline': ensemble_results['baseline'],
        'execution_time': execution_time
    }

def get_dataloaders(train_dataset, test_dataset, batch_size, bootstrap=False):
    """Create dataloaders, with bootstrap sampling if enabled."""
    if bootstrap:
        # Bootstrap sampling
        n_samples = len(train_dataset)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=bootstrap_sampler
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_dataloader, test_dataloader

def get_model_predictions(model, dataloader, device):
    """Get model predictions on a dataset."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs, axis=0)

def evaluate_ensemble(all_preds, test_dataloader, num_classes):
    """Evaluate ensemble predictions."""
    # Stack predictions from all trees
    all_preds = np.stack(all_preds, axis=0)  # shape: (num_trees, samples, num_classes)
    
    # Get true labels
    true_labels = []
    for _, y_batch in test_dataloader:
        true_labels.append(y_batch.numpy())
    true_labels = np.concatenate(true_labels, axis=0)
    
    # Get ensemble predictions (majority vote)
    argmax_preds = np.argmax(all_preds, axis=2)  # shape: (num_trees, samples)
    final_predictions = stats.mode(argmax_preds, axis=0)[0].flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, final_predictions)
    
    # Calculate AUC
    avg_probs = np.mean(all_preds, axis=0)  # Average probabilities across trees
    if num_classes > 2:
        auc = roc_auc_score(true_labels, avg_probs, multi_class='ovr', average='macro')
    else:
        auc = roc_auc_score(true_labels, avg_probs[:, 1])
    
    # Calculate baseline (majority class)
    print(true_labels)
    print(stats.mode(true_labels, axis = 0))
    baseline = accuracy_score(true_labels, np.full(true_labels.shape, stats.mode(true_labels, axis = 0)[0]))
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'baseline': baseline
    }

