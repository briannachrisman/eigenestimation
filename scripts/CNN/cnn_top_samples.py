import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision
import gc
from tqdm import tqdm

# Import functions from eigenestimation
from eigenestimation.evaluation.top_logits import compute_jacobian
from eigenestimation.evaluation.top_images import compute_circuit_vals
from eigenestimation.toy_models.cnn_wrapper import ImageOnlyDataset

# Set path for eigenmodel

# Main script function
def main(args):
    # Load Eigenmodel
    eigenmodel_path = args.model_path
    eigenmodel = torch.load(eigenmodel_path)['model']
    eigenmodel.to(args.device)
    print("Loaded eigenmodel:", args.model_path)

    # Define transformations for CIFAR-100
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),  # Resize to match ResNet input
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load CIFAR-100 dataset
    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_tensor)

    random_indices = random.sample(range(len(dataset)), args.num_samples)
    X_raw = ImageOnlyDataset(Subset(dataset, random_indices))
    torch.save(torch.stack([(X_raw[i]) for i in range(len(X_raw))]), args.examples_output_file)

    X_train = (torch.stack([transform(X_raw[i]) for i in range(len(X_raw))]))  # Stack images


    # Randomly select subset of images
    del dataset, X_raw
    
    
    dataset_raw = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_tensor)

    # Create DataLoader
    dataloader = DataLoader(X_train, batch_size=args.batch_size, shuffle=False)

    print(f"Processing {args.num_samples} images from CIFAR-100 with batch size {args.batch_size}.")

    # Compute Circuit Values
    print("Starting circuit value computation...")
    circuit_vals, X_ordered = compute_circuit_vals(eigenmodel, dataloader, args.iters, jac_chunk_size=args.jac_chunk_size)
    print("Computed circuit values.")

    # Save results
    torch.save(circuit_vals, args.attributions_output_file)
    print("Saved results.")

# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Jacobian & Circuit Values for Image Model")

    parser.add_argument("--model-path", type=str, default="resnet-18-eigenmodel.pt", help="Eigenmodel filename")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of images to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for DataLoader")
    parser.add_argument("--iters", type=int, default=3, help="Iterations for computing circuit values")
    parser.add_argument("--jac-chunk-size", type=int, default=10, help="Chunk size for Jacobian computation")
    parser.add_argument("--attributions-output-file", type=str, default="circuit_attributions.pt", help="Output file for attributions")
    parser.add_argument("--examples-output-file", type=str, default="X_data.pt", help="Output file for examples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    args = parser.parse_args()
    main(args)
