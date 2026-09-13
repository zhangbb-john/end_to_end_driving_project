import os
import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True,
                        help='path to bench2drive root, e.g. data/bench2drive')
    parser.add_argument('--version', type=str, default='v1',
                        help='dataset version, default v1')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='ratio of validation routes')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--out', type=str, required=True,
                        help='output json path')
    args = parser.parse_args()

    version_dir = os.path.join(args.dataroot, args.version)
    assert os.path.isdir(version_dir), f'Not found: {version_dir}'

    # 收集所有合法 route
    all_routes = []
    for name in os.listdir(version_dir):
        if ('Town' in name and 'Route' in name and 'Weather' in name):
            all_routes.append(os.path.join(args.version, name))

    all_routes = sorted(all_routes)
    print(f'Total routes: {len(all_routes)}')

    # 随机划分
    random.seed(args.seed)
    random.shuffle(all_routes)

    val_num = int(len(all_routes) * args.val_ratio)
    val_routes = sorted(all_routes[:val_num])

    split = {
        'val': val_routes
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(split, f, indent=2)

    print(f'Val routes: {len(val_routes)}')
    print(f'Split file saved to: {args.out}')

if __name__ == '__main__':
    main()
