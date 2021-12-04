import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    with open(args.data_path, 'r') as reader:
        all_data = json.load(reader)['items']
    
    with open(args.indices_path, 'r') as reader:
        indices = json.load(reader)

    out_data = []
    for idx in indices:
        out_data.append(all_data[idx])
    
    test_data = {
        '_count_': len(out_data),
        'items': out_data
    }

    with open(args.output_path, 'w') as writer:
        json.dump(test_data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
