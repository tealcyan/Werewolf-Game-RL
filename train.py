import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, keys, values):
    d_k = query.size(-1)  # Dimension of the keys
    scores = torch.matmul(query, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))  # Corrected line
    weights = F.softmax(scores, dim=-1)  # Normalize the scores to get attention weights
    output = torch.matmul(weights, values)  # Weighted sum of the values
    return output, weights

# Example setup
torch.manual_seed(0)  # For reproducibility

# Small size embeddings for demonstration
dim = 10  # Smaller dimension for clarity
num_actions = 3  # Number of actions

# Create e_state and action embeddings
e_state = torch.randn(1, 1, dim)  # Query: [1, 1, dim]
action_embeddings = torch.randn(1, num_actions, dim)  # Keys and Values: [1, num_actions, dim]

# Apply scaled dot product attention
attended_output, attention_weights = scaled_dot_product_attention(e_state, action_embeddings, action_embeddings)

# Print the outputs
print("Attended Output:", attended_output)
print("Attention Weights:", attention_weights)
