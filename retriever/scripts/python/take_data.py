import json
import argparse
import logging

from libs.utils.logging import add_color_formater


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    add_color_formater(logging.root)
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    with open(args.data_path, 'r') as reader:
        all_data = json.load(reader)['items']

    logger.info("Loading indices...")
    with open(args.indices_path, 'r') as reader:
        indices = json.load(reader)

    logger.info("Taking data...")
    out_data = []
    for idx in indices:
        out_data.append(all_data[idx])
    
    test_data = {
        '_count_': len(out_data),
        'items': out_data
    }

    with open(args.output_path, 'w') as writer:
        json.dump(test_data, writer, indent=4, ensure_ascii=False)
    logger.info("Written taken data to `{}`".format(args.output_path))


if __name__ == "__main__":
    main()
