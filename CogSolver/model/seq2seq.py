# -*- coding:utf-8 -*-

from torch import nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, knowledge_base, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.knowledge_base = knowledge_base
        self.decoder = decoder

    def forward(self, input_variable, num_pos, input_lengths, span_length=None,
                target_variable=None, tree=None, max_length=None, beam_width=None):
        encoder_outputs, graph_adjs, encoder_hidden = self.encoder(
            input_var=input_variable,
            input_lengths=input_lengths,
            span_length=span_length,
            tree=tree,
            knowledge_base=self.knowledge_base
        )

        output = self.decoder(
            targets=target_variable,
            graph_adjs=graph_adjs,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            input_lengths=input_lengths,
            span_length=span_length,
            num_pos=num_pos,
            max_length=max_length,
            beam_width=beam_width
        )
        return output
