import json
import os
import logging
import unicodedata
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_DATA_PATH = "data/atd/AN TOAN_ALL_GUI THE.csv"
JSON_DATA_PATH = "data/atd/an_toan_dien.json"

ABBREVIATION_MAPPING = {
    'QT ATĐ': "an toàn điện",
    'PCCC': "phòng cháy chữa cháy",
    'QT-TBAKNT': "trạm biến áp không người trực",
    'TT-NĐ': None
}


def main():
    df = pd.read_csv(CSV_DATA_PATH, header=0)
    df = df[['CÂU HỎI + TRẢ LỜI', 'Ghi chú']]
    qa_pairs = []
    for _, row in df.iterrows():
        title = ABBREVIATION_MAPPING[row['Ghi chú']]
        if title is None:
            continue
        qa = row['CÂU HỎI + TRẢ LỜI']
        qa_inverse = qa[::-1]
        answer, question = qa_inverse.split(':', 1)
        answer = answer[::-1]
        question = question[::-1]
        if len(answer.split()) < 15:
            continue
        doc = {
            'question': [unicodedata.normalize('NFKC', question).lower()],
            'context': [{
                'title': unicodedata.normalize('NFKC', title).lower(),
                'text': unicodedata.normalize('NFKC', answer).lower()
            }]
        }
        qa_pairs.append(doc)
    
    with open(JSON_DATA_PATH, 'w') as writer:
        json.dump(qa_pairs, writer, indent=4, ensure_ascii=False)
    logger.info("Created {} question-context pairs".format(len(qa_pairs)))


if __name__ == "__main__":
    main()
