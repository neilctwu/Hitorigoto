from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from eval import GreedySearchDecoder, evaluateInput
from utils import loadPrepareData
from params import *

from torch import nn
import torch
import os

ckpt = 'ckpt/39900_checkpoint.tar'

datafile = os.path.join(corpus_name, "formatted_movie_lines.txt")
voc, _ = loadPrepareData(corpus_name, datafile, MAX_LENGTH)

checkpoint = torch.load(ckpt)
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']

# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)

encoder = encoder.to(device)
decoder = decoder.to(device)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)
