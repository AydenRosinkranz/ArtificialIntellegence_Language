## Bugs!!!!!!!!!!! No dataset named MTFraEng(batch_size=128) in my library????
import tensorflow as tf
import os
from d2l import tensorflow as d2l
os.chdir('ArtificialIntelligence_Language/Transformer')
exec(open('TransformerArchitecture.py').read())

data = d2l.MTFraEng(batch_size=128)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
key_size, query_size, value_size = 256, 256, 256
norm_shape = [2]
with d2l.try_gpu():
    encoder = TransformerEncoder(
        len(data.src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.0015)
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)