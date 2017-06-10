# Word Autocompletion Algorithm based on character-based LSTM

### Character-based LSTM
* idea from char-rnn (https://github.com/karpathy/char-rnn), some implementation from sherjilozair's tensorflow char-rnn implementation (https://github.com/sherjilozair/char-rnn-tensorflow)
* Korean language model (jamo based), beam search for k-top result
* Tested on MAC OS X, Ubuntu 14.04

### Demo for autocompletion
* prerequisites

```
# install requirements (anaconda, tensorflow, etc ...)
cd data
./download.sh # downloading dataset
```

* Train text generative model

```
python train.py
```

* Test sentence, word sampling

```
# Currently using pretrained models, changed hard-coded directories to use other model
python sample.py
```

* Run test web server

```
python main.py
open localhost:5000
```
