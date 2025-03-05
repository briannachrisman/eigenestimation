import argparse
import torch
from torch.utils.data import DataLoader
from eigenestimation.evaluation.top_tokens import load_eigenmodel, load_and_tokenize_dataset, compute_circuit_vals, extract_top_features

# Main script function
def main(args):
    eigenmodel, frac_activated = load_eigenmodel(args.model_path)
    tokenizer = eigenmodel.model.tokenizer
    eigenmodel.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("loaded tokenizer and model")
    X_transformer = load_and_tokenize_dataset(args.dataset, args.split, tokenizer, args.token_length, args.num_samples)
    print(X_transformer.device, 'X_transformer.device')
    dataloader = DataLoader(X_transformer, batch_size=args.batch_size, shuffle=True)
    print(torch.cuda.memory_allocated(), 'cuda memory allocated - before compute circuit vals')
    print("loaded data")
    circuit_vals, X_ordered = compute_circuit_vals(eigenmodel, dataloader, args.iters, jac_chunk_size=args.jac_chunk_size)
    print(torch.cuda.memory_allocated(), 'cuda memory allocated - after compute circuit vals')
    print("computed circuit vals")
    feature_data = extract_top_features(circuit_vals, X_ordered, tokenizer, frac_activated, args.token_length, args.top_k)
    print(torch.cuda.memory_allocated(), 'cuda memory allocated - after extract top features')
    print("extracted top tokens")
    # Save results to file
    torch.save(feature_data, args.attributions_output_file)
    torch.save(X_ordered, args.examples_output_file)
    print("saved results")
# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and analyze top features in a model")

    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="Dataset name")
    parser.add_argument("--split", type=str, default="train[:1%]", help="Train split")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the eigenmodel")
    parser.add_argument("--token-length", type=int, default=16, help="Token length for processing")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations for computing circuit values")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for dataloader")
    parser.add_argument("--attributions-output-file", type=str, default="circuit_attributions.pt", help="Output file")
    parser.add_argument("--examples-output-file", type=str, default="X_data.pt", help="Output file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tokens to extract")
    parser.add_argument("--jac-chunk-size", type=int, default=10, help="Chunk size for jacobian")
    args = parser.parse_args()
    main(args)