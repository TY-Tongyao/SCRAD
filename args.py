import argparse
import torch
import time

parser = argparse.ArgumentParser(description='SCRAD: Sequence Coherence-based Retrieval-Augmented Anomaly Detection')


parser.add_argument('--name', type=str, default="scrad_exp", help='Experiment name')
parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
parser.add_argument('--dataname', default='Reddit', type=str, 
                    choices=['Reddit', 'Question', 'Elliptic', 'Heal-Fraud', 'Amazon', 'Epinions'],
                    help='Dataset name')
parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Run mode')
parser.add_argument('--seed', default=42, type=int, help='Random seed')


parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (adjusted per dataset)')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay')


parser.add_argument('--seq_num', default=5, type=int, help='Number of sequences per node')
parser.add_argument('--seq_length', default=11, type=int, help='Length of each sequence (must be odd)')


parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden dimension for encoders')
parser.add_argument('--num_scales', default=3, type=int, help='Number of scales for macro encoding')
parser.add_argument('--gamma', default=0.1, type=float, help='Restart probability for macro propagation')


parser.add_argument('--k', default=5, type=int, help='Top-k sequences to retrieve')
parser.add_argument('--entropy_weight', default=1.0, type=float, help='Weight for entropy term in similarity')


parser.add_argument('--llm_model', default='deepseek', type=str, choices=['deepseek', 'llama'], help='LLM for coherence assessment')
parser.add_argument('--quantization', default=4, type=int, help='Bit quantization for LLM (4/8/16)')


parser.add_argument('--log', default=True, type=bool, help='Enable logging')
parser.add_argument('--save_dir', default='checkpoints', type=str, help='Directory for saving models')
parser.add_argument('--log_dir', default='logs', type=str, help='Directory for logs')

args = parser.parse_args()


args.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
args.name = f"{args.name}_{args.dataname}_{time.strftime('%Y%m%d-%H%M%S')}"


if args.seq_length % 2 == 0:
    args.seq_length += 1  
    print(f"Adjusted sequence length to {args.seq_length} (must be odd)")