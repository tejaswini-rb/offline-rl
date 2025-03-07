import torch
import numpy as np
from model import CQLAgent

def test_cql_loss():
    state_dim = 4
    action_dim = 1

    agent = CQLAgent(state_dim, action_dim)

    batch_size = 5
    states = torch.rand(batch_size, state_dim)
    actions = torch.rand(batch_size, action_dim) * 2 - 1  
    rewards = torch.rand(batch_size, 1)
    next_states = torch.rand(batch_size, state_dim)
    dones = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)

    loss = agent.get_q_loss(states, actions, rewards, next_states, dones)

    assert loss is not None, "CQL loss should not be None"
    assert torch.isfinite(loss).all(), "CQL loss should be a finite value"
    assert loss.item() > 0, "CQL loss should be positive"

    print(f"Test passed! CQL loss computed: {loss.item()}")

if __name__ == "__main__":
    test_cql_loss()
