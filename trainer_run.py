from __future__ import unicode_literals, print_function, division
from helpers import timeSince, wordIndexesFromSentence, showPlot
from main import EncoderRNN, AttnDecoderRNN, train, EOS_token, beamDecode
import random
import time
import torch
import torch.nn as nn
from torch import optim

from data_prep import prepareData
from nltk.translate.bleu_score import corpus_bleu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_char_token = 0
hidden_size = 256
char_embedding_dim = 25
char_representation_dim = 25
MAX_LENGTH = 10


def charIndexesFromSentence(char, sentence):
    indexes = [charIndexesFromWord(char, word) for word in sentence.split(' ')]
    indexes.append([EOS_char_token])
    return indexes

def charIndexesFromWord(char, word):
    return [char.char2index[character] for character in word]

# Prepare Data

input_lang, output_lang, input_char, train_pairs, test_pairs = prepareData('eng', 'fra', True)

print('\nCounting total pairs, train pairs, and test pairs...')
print('%d train pairs' % (len(train_pairs),))
print('%d test pairs' % (len(test_pairs),))
print('Example of a training pair: %s' % (train_pairs[0]))

print('\nTesting charIndexesFromSentence...')
sentence = 'je t aime'
print('Sentence: %s' % (sentence))
print('Char Indexes: %s' % (charIndexesFromSentence(input_char, sentence)))

def tensorFromSentence(lang, sentence):
    indexes = wordIndexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def evaluate(encoder, decoder, sentence, input_lang, input_char, max_length=MAX_LENGTH ):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        input_char_indexes = charIndexesFromSentence(input_char, sentence)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(torch.LongTensor([input_char_indexes[ei]]).to(device),
                                                     input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_hidden = encoder_hidden

        decoded_batch = beamDecode(decoder_hidden.unsqueeze(1), decoder, encoder_outputs.unsqueeze(1))

        return decoded_batch

def trainEpochs(encoder, decoder, n_epochs=20, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    # Train over n_epochs
    for epoch in range(n_epochs):

        # Print start of epoch
        print('Epoch %d' % (epoch))

        # Shuffle train_pairs
        random.shuffle(train_pairs)

        # Get tensors from pair
        training_pairs = [tensorsFromPair(pair) for pair in train_pairs]

        # Get character indexes
        training_char_indexes = [charIndexesFromSentence(input_char, pair[0]) for pair in train_pairs]

        # Train all train_pairs
        for i in range(1, len(train_pairs) + 1):
            # for i in range(1, 2):
            training_pair = training_pairs[i - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            input_char_indexes = training_char_indexes[i - 1]

            loss = train(input_tensor, target_tensor, input_char_indexes, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                timeSince(start, i / len(train_pairs)), i, i / len(train_pairs) * 100, print_loss_avg))

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)



encoder1 = EncoderRNN(input_char.n_chars, input_lang.n_words, hidden_size, char_embedding_dim, char_representation_dim).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
trainEpochs(encoder1, attn_decoder1)
hypotheses = []
list_of_references = []

for pair in test_pairs:
  # Evaluate pair
  print('>', pair[0])
  print('=', pair[1])
  decoded_batch = evaluate(encoder1, attn_decoder1, input_lang, input_char, pair[0])
  output_sentence = ' '.join([output_lang.index2word[index.item()] for index in decoded_batch[0][0][1:-1]])
  print('<', output_sentence)
  print('')

  # Append to corpus
  hypotheses.append(output_sentence.split(' '))
  list_of_references.append([pair[1].split(' ')])

# BLEU-1
weights = (1.0/1.0, )
score = corpus_bleu(list_of_references, hypotheses, weights)
print('BLEU-1: %.4f' % (score))

# BLEU-2
weights = (1.0/2.0, 1.0/2.0, )
score = corpus_bleu(list_of_references, hypotheses, weights)
print('BLEU-2: %.4f' % (score))

# BLEU-3
weights = (1.0/3.0, 1.0/3.0, 1.0/3.0, )
score = corpus_bleu(list_of_references, hypotheses, weights)
print('BLEU-3: %.4f' % (score))

# BLEU-4
weights = (1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, )
score = corpus_bleu(list_of_references, hypotheses, weights)
print('BLEU-4: %.4f' % (score))