
import torch 

def PrintTopActivatingTexts(eigenmodel, dataloader, feature_idx, top_n, bold_char='*', to_print=True):
    # Initialize an empty heap and a variable to keep track of the highest N JVPs
    top_jvps = []

    insert_char = eigenmodel.model.tokenizer.encode(bold_char)

    # Iterate through the dataloader
    for X, jacobian in dataloader:
        # Calculate the JVP for the current batch
        jvp = jvp = eigenmodel(jacobian)[...,feature_idx]##eigenmodel.jacobian_vector_product(jacobian, feature_idx)  # Dim X
        
        # For each X in the batch, compute the JVP and store the (JVP, X) pairs in the heap
        for x, jvp_vals in zip(X, jvp):
            for token_idx, jvp_val in enumerate(jvp_vals):
                # Use a tuple of (-jvp_val, x) to store in the heap (negative for max-heap behavior)
                top_jvps.append((jvp_val, x, token_idx))
            
        # Sort the list by jvp_val (descending order) and slice to keep the top N
        top_jvps.sort(key=lambda item: item[0], reverse=True)
                
        # Keep only the top N entries in the list
        top_jvps = top_jvps[:top_n]

    # After the loop, `top_jvps` contains the top N (jvp, X) pairs with the highest JVPs
    # To extract just the top N X values sorted by the JVP:
    top_n_X_values = [(x, token_idx, jvp_val.item()) for jvp_val,x, token_idx in top_jvps]

    if to_print:
        for x, token_idx, jvp_val in top_n_X_values:
            to_decode = x[:token_idx].cpu().detach().tolist() + insert_char + [x[token_idx]] + insert_char + x[(token_idx+1):].detach().cpu().tolist()
            text = eigenmodel.model.tokenizer.decode(to_decode)
            print(text, '->', jvp_val)
    return top_n_X_values

def PrintTopActivatingTextsAllFeatures(eigenmodel, dataloader, top_n, bold_char='*', to_print=True):

    insert_char = eigenmodel.model.tokenizer.encode(bold_char)
    top_n_X_values = {i:[] for i in range(eigenmodel.n_features)}
    # Iterate through the dataloader
    for X, jacobian in dataloader:
        # Calculate the JVP for the current batch
        jvp = eigenmodel(jacobian)  # Batch x features
        
        # For each X in the batch, compute the JVP and store the (JVP, X) pairs in the heap
        for feature_idx in range(jvp.shape[-1]):
            top_jvps = []
            for x, jvp_vals in zip(X, jvp[...,feature_idx]):
                for token_idx, jvp_val in enumerate(jvp_vals):
                    # Use a tuple of (-jvp_val, x) to store in the heap (negative for max-heap behavior)
                    top_n_X_values[feature_idx].append((jvp_val.item(), x, token_idx))
                
                # Sort the list by jvp_val (descending order) and slice to keep the top N
                top_n_X_values[feature_idx].sort(key=lambda item: item[0], reverse=True)
                        
                # Keep only the top N entries in the list
                top_n_X_values[feature_idx] = top_n_X_values[feature_idx][:top_n]

    if to_print:
        for feature_idx in range(len(top_n_X_values)):
            print(f'------feature {feature_idx}-------')
            for jvp_val, x, token_idx in top_n_X_values[feature_idx]:
                to_decode = x[:token_idx].cpu().detach().tolist() + insert_char + [x[token_idx]] + insert_char + x[(token_idx+1):].detach().cpu().tolist()
                text = eigenmodel.model.tokenizer.decode(to_decode)
                print(text.replace('\n', 'newline'), '->', jvp_val)
    return top_n_X_values



