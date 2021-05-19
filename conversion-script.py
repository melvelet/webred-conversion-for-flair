import tensorflow as tf
import csv


def read_examples(*dataset_paths):
    examples = []
    dataset = tf.data.TFRecordDataset(dataset_paths)
    for raw_sentence in dataset:
        # print(raw_sentence.numpy().decode('UTF-8'))
        sentence = tf.train.Example()
        sentence.ParseFromString(raw_sentence.numpy())
        examples.append(sentence)
    return examples


for dataset_name in ("webred_21", "webred_5"):
    path_to_webred = f"{dataset_name}.tfrecord"          # Path to where the WebRED data was downloaded.
    webred_sentences = read_examples(path_to_webred)
    # sentence = webred_sentences[2]  # As an instance of `tf.Example`.

    lines = []
    with open(f'{dataset_name}.TXT', 'w', encoding="utf-8", newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t')
        for sentence_no, sentence in enumerate(webred_sentences):
            tsvfile.write(f"\n# doc {sentence_no}\n")
            relation_name = sentence.features.feature['relation_name'].bytes_list.value[0].decode('utf-8')
            relation_name = relation_name.replace("'", "\\\'")
            source_name = sentence.features.feature['source_name'].bytes_list.value[0].decode('utf-8')
            target_name = sentence.features.feature['target_name'].bytes_list.value[0].decode('utf-8')
            sentence_str = sentence.features.feature['sentence'].bytes_list.value[0].decode('utf-8')

            temp_sent = ""
            # if sentence_no in [67]:
            #     print(sentence)
            # print(sentence_str)

            for i in range(len(sentence_str)):
                if sentence_str[i] == '.':
                    if (len(sentence_str) > i + 1 and sentence_str[i+1] == ' ') or len(sentence_str) == i + 1:
                        temp_sent += ' '
                temp_sent += sentence_str[i]


            sentence_str = temp_sent
            sentence_str = sentence_str.replace('* ', '').replace(',', ' ,').replace('?', ' ?') \
                .replace('!', ' !').replace(':', ' :').replace(';', ' ;').replace('(', '( ').replace('[', '[ ') \
                .replace(')', ' )').replace(']', ' ]').replace('}', '} ') \
                .replace('  ', ' ')
            sentence_list = sentence_str.split()

            subj_pos = -1
            obj_pos = -1
            multi_token_entity = ""
            entity_list = []
            sentence_list_processed = []
            current_entity = []
            for token_no, token in enumerate(sentence_list):
                if multi_token_entity:
                    entity = multi_token_entity
                    current_entity.append(token)
                else:
                    entity = 'O'

                if token.startswith('SUBJ{'):
                    entity = 'B-SUBJ'
                    current_entity.append(token)
                    if not token.endswith('}'):
                        multi_token_entity = "I-SUBJ"
                elif token.startswith('OBJ{'):
                    entity = 'B-OBJ'
                    current_entity.append(token)
                    if not token.endswith('}'):
                        multi_token_entity = "I-OBJ"

                if token.endswith('}'):
                    current_entity_str = " ".join(current_entity).replace('OBJ{', '').replace('SUBJ{', '').replace('}', '')
                    # print("source_name: '", source_name, "' target_name: '", target_name,
                    #       "' current_entity_str: '", current_entity_str, ',',
                    #       current_entity_str == source_name, current_entity_str == target_name)
                    if current_entity_str == source_name:
                        # if 'OBJ' not in entity:
                        #     print('True')
                        obj_pos = token_no
                    elif current_entity_str == target_name:
                        # if 'SUBJ' not in entity:
                        #     print('True')
                        subj_pos = token_no
                    multi_token_entity = None
                    current_entity = []

                token = token.replace('OBJ{', '').replace('SUBJ{', '').replace('}', '')

                sentence_list_processed.append(token)
                entity_list.append(entity)

            if obj_pos == -1:
                print(source_name, 'not found: ', sentence_str)
            if subj_pos == -1:
                print(target_name, 'not found', sentence_str)

            for i, val in enumerate(zip(sentence_list_processed, entity_list)):
                token, entity = val[0], val[1]
                row = [
                    i,
                    token,
                    entity,
                    f"['{relation_name}']" if i == obj_pos else "['N']",
                    f"[{subj_pos}]" if i == obj_pos else f"[{i}]"
                ]

                tsvwriter.writerow(row)
