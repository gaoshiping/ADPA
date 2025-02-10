import math
import random
from typing import Dict, List

import torch
from torch import Tensor


def softmax(x: List[float]) -> List[float]:
    """
    Computes the softmax of a list of floats.

    Args:
        x (List[float]): A list of floats.

    Returns:
        List[float]: Softmax probabilities corresponding to the input list.
    """
    if not x:
        return []

    # Compute the maximum value to prevent overflow
    max_val = max(x)

    # Compute the sum of exponentials
    sum_exp = sum(math.exp(i - max_val) for i in x)

    # Calculate softmax probabilities
    return [math.exp(i - max_val) / sum_exp for i in x]


def compress_logits(
        logits: torch.Tensor, top_k: int = 50, top_p: float = 0.99999
) -> List[Dict[str, List[float]]]:
    """
    Compresses a logits matrix by retaining the top_k logits and those contributing to top_p cumulative probability.

    Args:
        logits (torch.Tensor): Input logits matrix of shape (seq_len, vocab_size).
        top_k (int, optional): Maximum number of top logits to retain. Defaults to 50.
        top_p (float, optional): Cumulative probability threshold. Defaults to 0.99999.

    Returns:
        List[Dict[str, List[float]]]: Compressed logits as a list of dictionaries containing indices and values.
    """
    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get top_k probabilities and their indices
    top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1)

    # Sort the top_k probabilities in descending order
    sorted_values, sorted_indices = torch.sort(top_k_values, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_values, dim=-1)

    # Create a mask for probabilities within the top_p threshold
    mask = cumulative_probs <= top_p

    # Ensure at least one token is included by adding the next token
    true_counts = mask.sum(dim=-1, keepdim=True)
    next_indices = torch.clamp(true_counts, max=top_k - 1)
    mask.scatter_(1, next_indices, True)

    # Generate the compressed logits list
    compressed_logits = []
    for i in range(logits.size(0)):
        valid_indices = sorted_indices[i][mask[i]]
        compressed_logits.append({
            'indices': top_k_indices[i][valid_indices].tolist(),
            'values': logits[i][top_k_indices[i][valid_indices]].tolist()  # Preserve original logits
        })

    return compressed_logits


def compress_probs(
        logits: torch.Tensor, top_k: int = 50, vocab_size: int = 32000
) -> List[Dict[str, List[float]]]:
    """
    Compresses a logits matrix by retaining the top_k probabilities and summing the remaining probabilities.

    Args:
        logits (torch.Tensor): Input logits matrix of shape (seq_len, vocab_size).
        top_k (int, optional): Maximum number of top probabilities to retain. Defaults to 50.
        vocab_size (int, optional): Size of the vocabulary. Defaults to 32000.

    Returns:
        List[Dict[str, List[float]]]: Compressed probabilities as a list of dictionaries containing indices, values, and remaining probability sum.
    """
    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get top_k probabilities and their indices
    top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1)

    # Calculate the sum of the remaining probabilities
    remaining_probs_sum = 1 - top_k_values.sum(dim=-1)

    # Generate the compressed probabilities list
    compressed_probs = []
    for i in range(logits.size(0)):
        compressed_probs.append({
            'indices': top_k_indices[i].tolist(),
            'values': top_k_values[i].tolist(),  # Preserve probability values
            'remaining_probs_sum': remaining_probs_sum[i].item()  # Sum of remaining probabilities
        })
    return compressed_probs


def recover_margin_cd(
        compressed_probs: List[List[Dict]],
        input_probs: Tensor,
        margin_logp_every: List[List[Dict]],
        threshold_p: float = 0.1,
        do_sample: bool = True,
        mini_vocab_size: int = 50,
        use_softmax: bool = True,
) -> List[List[Dict]]:
    """
    Recovers margin conditional distributions based on compressed probabilities and margin log-probabilities.

    Args:
        compressed_probs (List[List[Dict]]): Compressed probabilities.
        input_probs (Tensor): Input probabilities tensor.
        margin_logp_every (List[List[Dict]]): Margin log-probabilities for each token.
        threshold_p (float, optional): Probability threshold. Defaults to 0.1.
        do_sample (bool, optional): Whether to perform sampling. Defaults to True.
        mini_vocab_size (int, optional): Minimum vocabulary size to maintain. Defaults to 50.
        use_softmax (bool, optional): Whether to apply softmax to margin log-probabilities. Defaults to True.

    Returns:
        List[List[Dict]]: Recovered margin conditional distributions.
    """
    margin_cd = []
    batch_size, seq_len, vocab_size = input_probs.shape

    for batch_idx in range(batch_size):
        margin_cd_batch = []
        current_seq_len = len(compressed_probs[batch_idx])

        for seq_idx in range(current_seq_len):
            # Extract compressed probability and margin log-probability for the current position
            probs_position = compressed_probs[batch_idx][seq_idx]
            margin_logp_position = margin_logp_every[batch_idx][seq_idx]

            # Step 1: Select tokens with probability >= threshold_p
            v_head_tokens = [
                vocab_idx
                for vocab_idx, p in zip(probs_position['indices'], probs_position['values'])
                if p >= threshold_p
            ]

            # If no tokens meet the threshold, relax the threshold progressively
            if not v_head_tokens:
                v_head_tokens = [
                    vocab_idx
                    for vocab_idx, p in zip(probs_position['indices'], probs_position['values'])
                    if p >= threshold_p * 0.1
                ]

                if not v_head_tokens:
                    v_head_tokens = [
                        vocab_idx
                        for vocab_idx, p in zip(probs_position['indices'], probs_position['values'])
                        if p >= threshold_p * 0.01
                    ]

            # Limit the number of head tokens to mini_vocab_size
            if len(v_head_tokens) > mini_vocab_size:
                v_head_tokens = sorted(
                    zip(probs_position['indices'], probs_position['values']),
                    key=lambda x: x[1],
                    reverse=True
                )[:mini_vocab_size]
                v_head_tokens = [token for token, _ in v_head_tokens]

            # Step 2: Retrieve margin log-probabilities for the head tokens
            v_head_indices = [
                i for i, vocab_idx in enumerate(margin_logp_position['indices'])
                if vocab_idx in v_head_tokens
            ]
            v_head_mlogp = [
                margin_logp_position['values'][i] for i in v_head_indices
            ]

            # Step 3: Sort tokens based on margin log-probabilities
            v_head_token_mlogp = sorted(
                zip(
                    [margin_logp_position['indices'][j] for j in v_head_indices],
                    v_head_mlogp
                ),
                key=lambda x: x[1],
                reverse=True
            )
            sorted_tokens, sorted_mlogp = zip(*v_head_token_mlogp) if v_head_token_mlogp else ([], [])

            # Apply softmax if required
            if use_softmax and sorted_mlogp:
                sorted_mlogp = softmax(sorted_mlogp)
            else:
                sorted_mlogp = list(sorted_mlogp)

            # Step 4: Construct the margin conditional distribution
            if not do_sample:
                # Assign probability 1.0 to the top token
                if sorted_tokens:
                    margin_cd_seq = {
                        "indices": sorted_tokens[0],
                        "values": 1.0,
                        "remaining_probs_sum": 0.0
                    }
                else:
                    margin_cd_seq = {
                        "indices": [],
                        "values": [],
                        "remaining_probs_sum": 0.0
                    }

            elif len(sorted_tokens) <= mini_vocab_size:
                # Sample tokens and pad if necessary
                pad_size = mini_vocab_size - len(sorted_tokens)
                pad_token_idx = random.sample(
                    list(set(range(vocab_size)) - set(v_head_tokens)),
                    pad_size
                ) if pad_size > 0 else []
                margin_cd_seq = {
                    "indices": list(sorted_tokens) + pad_token_idx,
                    "values": list(sorted_mlogp) + [0.0] * len(pad_token_idx),
                    "remaining_probs_sum": 0.0
                }
            else:
                # Assign probability 1.0 to the top token
                if sorted_tokens:
                    margin_cd_seq = {
                        "indices": sorted_tokens[0],
                        "values": 1.0,
                        "remaining_probs_sum": 0.0
                    }
                else:
                    margin_cd_seq = {
                        "indices": [],
                        "values": [],
                        "remaining_probs_sum": 0.0
                    }

            margin_cd_batch.append(margin_cd_seq)
        margin_cd.append(margin_cd_batch)

    return margin_cd


def load_input_and_target_probs_fast(
        compressed_probs: List[List[Dict]],
        input_probs: Tensor,
        labels: Tensor,  # Shape: [batch_size, seq_len]
        from_chosen_or_rejected: str = None,
        prob_stabilize: bool = True,
        atkd: bool = False
) -> Tensor:
    """
    Loads and processes input and target probabilities based on compressed probabilities and labels.

    Args:
        compressed_probs (List[List[Dict]]): Compressed probabilities.
        input_probs (Tensor): Input probabilities tensor.
        labels (Tensor): Labels tensor of shape [batch_size, seq_len].
        from_chosen_or_rejected (str, optional): Mode of adjustment ('chosen' or 'rejected'). Defaults to None.
        prob_stabilize (bool, optional): Whether to stabilize probabilities. Defaults to True.
        atkd (bool, optional): Additional processing flag. Defaults to False.

    Returns:
        Tensor: Processed output and target probabilities, and an optional mask.
    """
    device = labels.device
    batch_size, seq_len, vocab_size = input_probs.shape

    # Extend input_probs with an extra dimension for remaining probabilities
    input_probs_extended = torch.cat(
        [input_probs, torch.zeros(batch_size, seq_len, 1, device=device)],
        dim=-1
    )

    min_vocab_size = len(compressed_probs[0][0]['indices'])
    target_probs = torch.zeros([batch_size, seq_len, min_vocab_size + 1], device=device)
    output_probs = torch.zeros([batch_size, seq_len, min_vocab_size + 1], device=device)

    # Adjust compressed_probs based on chosen or rejected tokens
    if from_chosen_or_rejected:
        compressed_probs_new_all = compressed_probs.copy()
        for i in range(len(compressed_probs)):
            for j in range(len(compressed_probs[i])):
                compressed = compressed_probs[i][j]
                label_start = torch.where(labels[i] != -100)[0][0].item()
                label = labels[i][label_start + j].item()

                if from_chosen_or_rejected == "chosen":
                    if compressed['indices'][0] != label:
                        chosen_vocab_index = label
                        try:
                            original_index = compressed['indices'].index(chosen_vocab_index)
                            rank1_p = compressed['values'][0]
                            decrease_ratio = (1 - rank1_p) / (1 - compressed['values'][original_index])
                            out_indices = [idx for idx in range(len(compressed['indices'])) if
                                           compressed['indices'][idx] != chosen_vocab_index]
                        except ValueError:
                            out_indices = []
                            decrease_ratio = 1 - rank1_p

                        # Limit to min_vocab_size
                        if len(out_indices) >= min_vocab_size:
                            out_indices = out_indices[:min_vocab_size - 1]
                            compressed["remaining_probs_sum"] += sum(compressed['values'][min_vocab_size - 1:])

                        # Update compressed_probs with the chosen token
                        compressed_probs_new = {
                            "indices": [chosen_vocab_index] + [compressed['indices'][idx] for idx in out_indices],
                            "values": [rank1_p] + [compressed['values'][idx] * decrease_ratio for idx in out_indices],
                            "remaining_probs_sum": compressed["remaining_probs_sum"] * decrease_ratio
                        }
                        compressed_probs_new_all[i][j] = compressed_probs_new

                elif from_chosen_or_rejected == "rejected":
                    if label in compressed['indices']:
                        rejected_index = compressed['indices'].index(label)
                        increase_ratio = 1 / max(0.01, 1 - compressed['values'][rejected_index])
                        out_rejected_indices = [
                            idx for idx in range(len(compressed['indices'])) if compressed['indices'][idx] != label
                        ]
                        compressed_probs_new = {
                            "indices": [label] + [compressed['indices'][idx] for idx in out_rejected_indices],
                            "values": [0.0] + [compressed['values'][idx] * increase_ratio for idx in
                                               out_rejected_indices],
                            "remaining_probs_sum": compressed["remaining_probs_sum"] * increase_ratio
                        }
                        compressed_probs_new_all[i][j] = compressed_probs_new

        compressed_probs = compressed_probs_new_all

    # Additional processing if atkd is True
    easy_xy = []
    if atkd:
        compressed_probs_new_all = compressed_probs.copy()
        for i in range(len(compressed_probs)):
            unc_list = []
            for j in range(len(compressed_probs[i])):
                compressed = compressed_probs[i][j]
                label_start = torch.where(labels[i] != -100)[0][0].item()
                if len(labels[i]) <= label_start + j:
                    continue
                label = labels[i][label_start + j].item()

                # Calculate uncertainty
                if label in compressed['indices']:
                    label_index = compressed['indices'].index(label)
                    unc = 1 - compressed['values'][label_index]
                else:
                    unc = 1.0
                unc_list.append(unc)

            # Sort and select easy examples
            sorted_unc = sorted(enumerate(unc_list), key=lambda x: x[1])
            easy_set = [idx for idx, _ in sorted_unc[:len(sorted_unc) // 2]]
            easy_xy.extend([(i, k) for k in easy_set])

            # Reconstruct probability distribution for easy examples
            for j in easy_set:
                compressed = compressed_probs[i][j]
                label_start = torch.where(labels[i] != -100)[0][0].item()
                if len(labels[i]) <= label_start + j:
                    continue
                label = labels[i][label_start + j].item()
                if label in compressed['indices']:
                    label_idx = compressed['indices'].index(label)
                    remaining = compressed["remaining_probs_sum"] + compressed['values'][label_idx]
                    new_indices = [k for k in compressed['indices'] if k != label]
                    new_values = [v for idx, v in enumerate(compressed['values']) if idx != label_idx]
                    new_indices.append(random.choice(list(set(range(vocab_size)) - set(compressed['indices']))))
                    new_values.append(0.0)
                    compressed_probs_new_all[i][j] = {
                        "indices": new_indices,
                        "values": new_values,
                        "remaining_probs_sum": remaining
                    }

        compressed_probs = compressed_probs_new_all

    # Process each batch to construct target and output probabilities
    for i in range(batch_size):
        # Skip if no valid labels
        if not torch.any(labels[i] != -100):
            continue

        # Extract compressed values and indices
        compressed_values = torch.tensor(
            [pos['values'] for pos in compressed_probs[i]],
            device=device
        )
        compressed_remaining = torch.tensor(
            [pos.get("remaining_probs_sum", 0.0) for pos in compressed_probs[i]],
            device=device
        )
        compressed_indices = torch.tensor(
            [pos['indices'] for pos in compressed_probs[i]],
            device=device
        )

        # Handle out-of-vocab indices by adding their probabilities to remaining_probs_sum
        out_vocab_mask = compressed_indices >= vocab_size
        masked_floats = torch.where(out_vocab_mask, compressed_values, torch.tensor(0.0, device=device))
        compressed_remaining += masked_floats.sum(dim=1)

        # Set out-of-vocab indices and values to a special index
        compressed_indices = torch.where(out_vocab_mask, torch.tensor(vocab_size, device=device), compressed_indices)
        compressed_values = torch.where(out_vocab_mask, torch.tensor(0.0, device=device), compressed_values)

        # Determine label boundaries
        valid_label_indices = torch.where(labels[i] != -100)[0]
        if valid_label_indices.numel() == 0:
            continue
        l, r = valid_label_indices.min().item(), valid_label_indices.max().item()
        response_length = compressed_values[:r + 1 - l].shape[0]

        # Assign target probabilities
        target_probs[i, l: l + response_length, :min_vocab_size] += compressed_values[:r + 1 - l]
        target_probs[i, l: l + response_length, min_vocab_size] += compressed_remaining[:r + 1 - l]

        # Assign output probabilities
        gathered_probs = torch.gather(input_probs_extended[i, l: l + response_length], 1, compressed_indices[:r + 1 - l].long())
        output_probs[i, l: l + response_length, :min_vocab_size] = gathered_probs
        output_probs[i, l: l + response_length, min_vocab_size] = 1 - output_probs[i, l: l + response_length, :min_vocab_size].sum(-1)

    # Stabilize probabilities to avoid numerical issues
    if prob_stabilize:
        output_probs = torch.clamp(output_probs, min=1e-8, max=1 - 1e-8)
        target_probs = torch.clamp(target_probs, min=1e-8, max=1 - 1e-8)

    # Create an easy mask if atkd is enabled
    if atkd:
        if not easy_xy:
            is_easy_mask = torch.full_like(target_probs, fill_value=False, dtype=torch.bool)
        else:
            x_coords, y_coords = zip(*easy_xy)
            is_easy_mask = torch.full_like(target_probs, fill_value=False, dtype=torch.bool)
            is_easy_mask[torch.tensor(x_coords), torch.tensor(y_coords)] = True
        return output_probs, target_probs, is_easy_mask

    return output_probs, target_probs


def restore_logits(
        compressed_logits: List[List[Dict[str, List[float]]]],
        mask_labels: Tensor,
        batch_size: int,
        seq_len: int,
        vocab_size: int
) -> Tensor:
    """
    Restores the full logits tensor from compressed logits.

    Args:
        compressed_logits (List[List[Dict[str, List[float]]]]): Compressed logits.
        mask_labels (Tensor): Mask labels tensor.
        batch_size (int): Number of batches.
        seq_len (int): Sequence length.
        vocab_size (int): Vocabulary size.

    Returns:
        Tensor: Restored logits tensor of shape (batch_size, seq_len, vocab_size).
    """
    # Initialize logits with -inf
    restored_logits = torch.full((batch_size, seq_len, vocab_size), -torch.inf)

    for batch_id, (batch, mask) in enumerate(zip(compressed_logits, mask_labels)):
        j = 0  # Index for compressed logits
        for i, mask_token in enumerate(mask):
            if mask_token == -100:
                restored_logits[batch_id, i, :] = 0
            elif j >= len(batch):
                j += 1
                continue
            else:
                indices = batch[j]['indices']
                values = batch[j]['values']
                try:
                    restored_logits[batch_id, i, indices] = torch.tensor(values)
                except IndexError:
                    # Handle out-of-vocab indices by ignoring them
                    valid_indices = [idx for idx in indices if 0 <= idx < vocab_size]
                    valid_values = [values[k] for k, idx in enumerate(indices) if 0 <= idx < vocab_size]
                    restored_logits[batch_id, i, valid_indices] = torch.tensor(valid_values)
                j += 1

    return restored_logits.to(torch.bfloat16)


def main():
    """
    Main function to test all components of the module.
    """
    # Test softmax function
    print("Testing softmax function:")
    sample_input = [1.0, 2.0, 3.0]
    print(f"Input: {sample_input}")
    print(f"Softmax Output: {softmax(sample_input)}\n")

    # Create sample logits tensor
    batch_size = 2
    seq_len = 3
    vocab_size = 10
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test compress_logits function
    print("Testing compress_logits function:")
    compressed_logits = compress_logits(logits[0], top_k=5, top_p=0.9)
    print(f"Compressed Logits: {compressed_logits}\n")

    # Test compress_probs function
    print("Testing compress_probs function:")
    compressed_probs = compress_probs(logits[0], top_k=5, vocab_size=vocab_size)
    print(f"Compressed Probs: {compressed_probs}\n")

    # Create sample margin_logp_every
    margin_logp_every = [
        [
            {'indices': list(range(vocab_size)), 'values': list(torch.randn(vocab_size).numpy())}
            for _ in range(seq_len)
        ]
        for _ in range(batch_size)
    ]

    # Test recover_margin_cd function
    print("Testing recover_margin_cd function:")
    recovered_margin_cd = recover_margin_cd(
        compressed_probs=[compressed_probs] * batch_size,
        input_probs=logits,
        margin_logp_every=margin_logp_every,
        threshold_p=0.1,
        do_sample=True,
        mini_vocab_size=5,
        use_softmax=True
    )
    print(f"Recovered Margin CD: {recovered_margin_cd}\n")

    # Create sample labels tensor
    labels = torch.tensor([
        [1, 2, 3],
        [4, 5, -100]
    ])

    # Test load_input_and_target_probs_fast function
    print("Testing load_input_and_target_probs_fast function:")
    output_probs, target_probs = load_input_and_target_probs_fast(
        compressed_probs=[compressed_probs] * batch_size,
        input_probs=logits,
        labels=labels,
        from_chosen_or_rejected=None,
        prob_stabilize=True,
        atkd=False
    )
    print(f"Output Probs: {output_probs}")
    print(f"Target Probs: {target_probs}\n")

    # Test restore_logits function
    print("Testing restore_logits function:")
    mask_labels = labels
    restored = restore_logits(
        compressed_logits=[compressed_logits] * batch_size,
        mask_labels=mask_labels,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    print(f"Restored Logits: {restored}\n")


if __name__ == "__main__":
    main()
