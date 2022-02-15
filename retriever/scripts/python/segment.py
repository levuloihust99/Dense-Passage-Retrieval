import json
import argparse
import logging

from libs.data_helpers.segment import segment_recursive
from libs.utils.logging import add_color_formater


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--segmenter-path",
                        default="vncorenlp/VnCoreNLP-1.1.1.jar")
    parser.add_argument(
        "--segment-keys", default=None)
    parser.add_argument("--debug", action='store_const',
                        const=True, default=False)
    parser.add_argument("--segment-host", required=True)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging.INFO)
    add_color_formater(logging.root)
    logger = logging.getLogger(__name__)

    logger.info("Loading json data...")
    with open(args.input_file, 'r') as reader:
        data = json.load(reader)
    logger.info("Instantiating RDR segmenter...")
    segment_keys = set(args.segment_keys.split(',')) if args.segment_keys else args.segment_keys

    logger.info("Segmenting data...")
    data = segment_recursive(data, args.segment_host, segment_keys)
    with open(args.output_file, 'w') as writer:
        json.dump(data, writer, indent=4, ensure_ascii=False)
    logger.info("Written segmented data to `{}`".format(args.output_file))


if __name__ == "__main__":
    main()
