---
title: Seq2Seq with GRU
slug: /seq2seq-GRU
---

The [sequence to sequence model](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) originates from language translation. Our implementation adapts the model for multi-step time series forecasting.  Specifically, given the input series $x_1,\cdots, x_{t}$, the model maps the input series to output series

$ x_{t-p}, \cdots, x_{t-1} \longrightarrow x_t, x_{t+1}, \cdots, x_{t+h-1}$

where $p$ is the input history length and $h$ is forecasting horizon.

Sequence to sequence (Seq2Seq) model consists of an encoder and a decoder. The final state of the encoder is fed as the initial state of the decoder. For both the encoder and the decoder, we can use various modules. This function implements  Gated Recurrent Unit (GRU).
