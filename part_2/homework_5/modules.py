import random
import torch
from torch import nn
from torch.nn import functional as F
 
def softmax(x, temperature=5): # use your temperature
  e_x = torch.exp(x / temperature)
  return e_x / torch.sum(e_x, dim=0)
 
class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
      super().__init__()
      
      self.input_dim = input_dim
      self.emb_dim = emb_dim
      self.hid_dim = hid_dim
      self.n_layers = n_layers
      self.dropout = dropout
      self.bidirectional = bidirectional
      
      self.embedding = nn.Embedding(input_dim, emb_dim)
      
      self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
      
      self.dropout = nn.Dropout(p=dropout)
      
  def forward(self, src):
      
      embedded = self.dropout(self.embedding(src))
      
      outputs, (hidden, cell) = self.rnn(embedded)
          
      return outputs, hidden, cell
 
 
class Attention(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
      super().__init__()
      
      self.enc_hid_dim = enc_hid_dim
      self.dec_hid_dim = dec_hid_dim
      
      self.attn = nn.Linear(enc_hid_dim + dec_hid_dim*2, dec_hid_dim)
      self.v = nn.Linear(dec_hid_dim, 1)
      
  def forward(self, hidden, encoder_outputs):

      '''your code'''
      repeated_hidden = hidden[-1].detach().clone().repeat(encoder_outputs.size()[0], 1, 1)
      encoder_outputs = torch.cat([encoder_outputs, repeated_hidden], dim=2)
      attn = self.attn(encoder_outputs)
      '''your code'''
      
      energy = self.v(torch.tanh(attn))
      # print("energy => ", energy.shape)
 
      # get attention, use softmax function which is defined, can change temperature
      '''your code'''
      softmax_energy = softmax(energy)
          
      return softmax_energy
  
  
class DecoderWithAttention(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
      super().__init__()
 
      self.emb_dim = emb_dim
      self.enc_hid_dim = enc_hid_dim
      self.dec_hid_dim = dec_hid_dim
      self.output_dim = output_dim
      self.attention = attention
      
      self.embedding = nn.Embedding(output_dim, emb_dim)
      
      self.rnn = nn.GRU(enc_hid_dim*2 + emb_dim + dec_hid_dim, dec_hid_dim, )
      
      self.out = nn.Linear(enc_hid_dim*2 + dec_hid_dim + emb_dim, output_dim) # linear layer to get next word
      
      self.dropout = nn.Dropout(dropout)
      
  def forward(self, input, hidden, encoder_outputs):
      
      input = input.unsqueeze(0) # because only one word, no words sequence 
      
      embedded = self.dropout(self.embedding(input))
      if torch.tensor(embedded.shape).shape[0] > 3:
        embedded = embedded.squeeze(0)
      
      #embedded = [1, batch size, emb dim]
      
      # get weighted sum of encoder_outputs
      '''your code'''
      weighted_sum = self.attention(hidden, encoder_outputs)
      '''your code'''
      weighted_sum = torch.matmul(weighted_sum.T.permute(1, 0, 2), encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
      output, hidden = self.rnn(torch.cat([embedded, weighted_sum, hidden[-1].unsqueeze(0)], dim=-1))
      # get predictions
      '''your code'''
      predictions = self.out(torch.cat([embedded, hidden, weighted_sum], dim = -1))
      
      # return '''your code'''
      return predictions, hidden
      
 
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
      super().__init__()
      
      self.encoder = encoder
      self.decoder = decoder
      self.device = device
      
      assert encoder.hid_dim == decoder.dec_hid_dim, \
          "Hidden dimensions of encoder and decoder must be equal!"
      
  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
      
      # src = [src sent len, batch size]
      # trg = [trg sent len, batch size]
      # teacher_forcing_ratio is probability to use teacher forcing
      # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
      
      # Again, now batch is the first dimention instead of zero
      batch_size = trg.shape[1]
      trg_len = trg.shape[0]
      trg_vocab_size = self.decoder.output_dim
      
      #tensor to store decoder outputs
      outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
      
      #last hidden state of the encoder is used as the initial hidden state of the decoder
      # print("SEQ2SEQ INPUT ==>", src.shape)
      enc_states, hidden, cell = self.encoder(src)
      # print("ENCODER OUTPUT ==>", enc_states.shape, hidden.shape)
      
      #first input to the decoder is the <sos> tokens
      input = trg[0,:]
      
      for t in range(1, trg_len):
 
          '''your code'''
          # print("seq2seq hidden => ", hidden.shape)
          output, hidden = self.decoder(input, hidden, enc_states)
          outputs[t] = output
          #decide if we are going to use teacher forcing or not
          teacher_force = random.random() < teacher_forcing_ratio
          #get the highest predicted token from our predictions
          top1 = output.argmax(-1) 
          #if teacher forcing, use actual next token as next input
          #if not, use predicted token
          input = trg[t] if teacher_force else top1
      
      return outputs