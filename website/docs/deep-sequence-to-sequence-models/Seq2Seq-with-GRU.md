---
title: Seq2Seq with GRU
slug: /seq2seq-gru
---

The [sequence to sequence model](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) originates from language translation. Our implementation adapts the model for multi-step time series forecasting. Specifically, given the input series $x_1, \ldots, x_{t}$, the model maps the input series to the output series:

$$
x_{t-p}, x_{t-p+1}, \ldots, x_{t-1} \longrightarrow x_t, x_{t+1}, \ldots, x_{t+h-1}
$$

where $p$ is the input history length and $h$ is the forecasting horizon.

Sequence to sequence (Seq2Seq) models consist of an encoder and a decoder. The final state of the encoder is fed as the initial state of the decoder. We can use various models for both the encoder and decoder. This function implements a Gated Recurrent Unit (GRU).
