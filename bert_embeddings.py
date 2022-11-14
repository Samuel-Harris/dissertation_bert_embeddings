import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import math
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

from time import time
import cProfile

preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3')
bert = hub.load('https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4')

def generate_sentence_embedings(sentence, max_seq_length):
    # tokenise sentence
    tokenised_words = preprocessor.tokenize(tf.constant([sentence]))[0]
    sub_word_counts = list(map(len, tokenised_words))
    bert_inputs = preprocessor.bert_pack_inputs([tokenised_words], tf.constant(max_seq_length))

    # produce sub-word embeddings
    num_sub_words = sum(sub_word_counts)
    sub_word_embeddings = bert(bert_inputs)['encoder_outputs'][-2][0][:num_sub_words]
    
    # combine sub-word embeddings into word embeddings
    word_embeddings = []
    sub_word_embeddings_i = 0
    for num_subwords in sub_word_counts:
        sub_word_vectors = sub_word_embeddings[sub_word_embeddings_i:sub_word_embeddings_i+num_subwords]
        word_embeddings.append(tf.math.reduce_mean(sub_word_vectors, 0))
        sub_word_embeddings_i += num_subwords

    return tf.stack(word_embeddings)

def generate_and_save_embeddings_batch(data, output_embeddings_file_path, output_metadata_file_path, max_seq_length, start_i, batch_size, embedding_vector_size=768):
    sentence_embedding_list = []
    key_list = []

    # generate embeddings
    for i in range(start_i, start_i+batch_size):
        key, sentence = data[i]
        word_embeddings = generate_sentence_embedings(sentence, max_seq_length)
        sentence_embedding_list.append(word_embeddings)
        key_list.append(key)

    # put embeddings into an array
    max_len = max(map(len, sentence_embedding_list))
    sentence_embedding_arr = np.full((batch_size, max_len, embedding_vector_size), -1, dtype='float32')

    for i in range(batch_size):
        sentence_embedding_arr[i][:sentence_embedding_list[i].shape[0]] = sentence_embedding_list[i]

    # write embeddings array to file
    np.save(output_embeddings_file_path, sentence_embedding_arr)

    # write key list to file
    with open(output_metadata_file_path, 'w+') as f:
        f.write('\n'.join(key_list))

def generate_embeddings(input_file_path, output_embeddings_file_path, output_metadata_file_path, max_seq_length=512, batch_size=1000):
    with open(input_file_path, 'r') as dataset_f:
        data = [(key, sentence) for key, sentence in map(lambda line: line.split(':', 1), dataset_f.readlines())]
    
    len_data = len(data)
    num_batches = math.ceil(len_data / batch_size)

    for batch_num in range(len_data // batch_size):
        print('processing batch {}/{}...'.format(batch_num+1, num_batches))
        generate_and_save_embeddings_batch(data, output_embeddings_file_path + str(batch_num), output_metadata_file_path + str(batch_num) + '.txt', max_seq_length, batch_num, batch_size)
    
    print('processing batch {}/{}...'.format(num_batches, num_batches))
    leftover = len_data % batch_size
    if leftover > 0:
        batch_num = num_batches-1
        generate_and_save_embeddings_batch(data, output_embeddings_file_path + str(batch_num), output_metadata_file_path + str(batch_num) + '.txt', max_seq_length, batch_num*batch_size, leftover)

input_file_path = 'output.txt'
output_embeddings_file_path = 'embeddings_'
output_metadata_file_path = 'metadata_'
generate_embeddings(input_file_path, output_embeddings_file_path, output_metadata_file_path, batch_size=3)
