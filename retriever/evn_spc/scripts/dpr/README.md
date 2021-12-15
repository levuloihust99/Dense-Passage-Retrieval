## Steps
### Step 1. process_annotated_data.py
- Convert excel into json
- Output: `evn_spc_question_answer.json`, `missing_parents.json`
### Step 2. merge_with_missing_parent.py
- Merge successful json at Step 1 (`evn_spc_question_answer.json`) with manually-fixed data (`missing_parents_added.json`).
- Output: `evn_spc_question_answer_full.json`
### Step 3. preprocess_for_train.py
- Convert into multi-questions multi-contexts format.
- Output: `evn_spc_question_answer_list_format.json`
### Step 4. remove_indicating_words.py
- Remove heading-indicating words like "Chương", "Mục", "Điều", "I.", "1.", "a.", "a)"
- Lowercase
- Output: `evn_spc_question_answer_lowercase_no_indicatingwords.json`