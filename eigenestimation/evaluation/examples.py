import torch 

def TopActivatingSamples(eigenmodel, dataloader, top_n=5, feature_idxs=None, to_print=False):

    if feature_idxs is None: feature_idxs = range(eigenmodel.n_features)
    top_n_X_values = {i:[] for i in feature_idxs}
    
    # Iterate through the dataloader
    for X, jacobian in dataloader:
        # Calculate the JVP for the current batch
        jvp = eigenmodel(jacobian)  # Batch x features
        
        # For each X in the batch, compute the JVP and store the (JVP, X) pairs in the heap
        for feature_idx in feature_idxs:
            top_jvps = []
            for x, jvp_val in zip(X, jvp[...,feature_idx]):
                # Use a tuple of (-jvp_val, x) to store in the heap (negative for max-heap behavior)
                top_n_X_values[feature_idx].append((jvp_val.item(), x))
                
                # Sort the list by jvp_val (descending order) and slice to keep the top N
                top_n_X_values[feature_idx].sort(key=lambda item: item[0], reverse=True)
                        
                # Keep only the top N entries in the list
                top_n_X_values[feature_idx] = top_n_X_values[feature_idx][:top_n]

    for feature_idx in top_n_X_values:
        if to_print: print(f'------feature {feature_idx}-------')
        for i, (jvp_val, x) in enumerate(top_n_X_values[feature_idx]):
            if to_print: print(f'{x} -> {jvp_val}')
    return top_n_X_values



def TopActivatingTexts(eigenmodel, dataloader, top_n=5, feature_idxs=None, bold_char='*', to_print=False):

    insert_char = eigenmodel.model.tokenizer.encode(bold_char)
    if feature_idxs is None: feature_idxs = range(eigenmodel.n_features)
    top_n_X_values = {i:[] for i in feature_idxs}
    
    # Iterate through the dataloader
    for X, jacobian in dataloader:
        # Calculate the JVP for the current batch
        jvp = eigenmodel(jacobian)  # Batch x features
        
        # For each X in the batch, compute the JVP and store the (JVP, X) pairs in the heap
        for feature_idx in feature_idxs:
            top_jvps = []
            for x, jvp_vals in zip(X, jvp[...,feature_idx]):
                for token_idx, jvp_val in enumerate(jvp_vals):
                    # Use a tuple of (-jvp_val, x) to store in the heap (negative for max-heap behavior)
                    top_n_X_values[feature_idx].append((jvp_val.item(), x, token_idx))
                
                # Sort the list by jvp_val (descending order) and slice to keep the top N
                top_n_X_values[feature_idx].sort(key=lambda item: item[0], reverse=True)
                        
                # Keep only the top N entries in the list
                top_n_X_values[feature_idx] = top_n_X_values[feature_idx][:top_n]

    for feature_idx in range(len(top_n_X_values)):
        if to_print: print(f'------feature {feature_idx}-------')
        for i, (jvp_val, x, token_idx) in enumerate(top_n_X_values[feature_idx]):
            to_decode = x[:token_idx].cpu().detach().tolist() + insert_char + [x[token_idx]] + insert_char + x[(token_idx+1):].detach().cpu().tolist()
            text = eigenmodel.model.tokenizer.decode(to_decode).replace('\n', 'newline')
            if to_print: print('{text} -> {jvp_val}')
            top_n_X_values[feature_idx][i] = (jvp_val, x, token_idx, text)
    return top_n_X_values



