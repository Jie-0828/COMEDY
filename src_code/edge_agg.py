import torch
import torch.nn.functional as F

def edge_agg_function(edge_agg,embeds):
    # Convert to edge embedding
    if edge_agg == "mean":
        embeds = (embeds[0] + embeds[1]) / 2
        embeds = embeds.unsqueeze(1)
    # hadamard
    elif edge_agg == 'had':
        embeds = embeds[0].mul(embeds[1])
        embeds = embeds.unsqueeze(1)
    # weight-l1
    elif edge_agg == "w1":
        embeds = torch.abs(embeds[0] - embeds[1])
        embeds = embeds.unsqueeze(1)
    # weight-l2
    elif edge_agg == "w2":
        embeds = torch.abs(embeds[0] - embeds[1]).mul(torch.abs(embeds[0] - embeds[1]))
        embeds = embeds.unsqueeze(1)
    # activation
    elif edge_agg == 'activation':
        embeds = torch.cat((embeds[0], embeds[1]), 0).unsqueeze(1)
        embeds = F.relu(embeds)
    elif edge_agg == 'origin':
        embeds = torch.cat((embeds[0], embeds[1]), 0).unsqueeze(1)

    return embeds