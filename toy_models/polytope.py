class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


