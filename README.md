# Word Autocompletion Algorithm based on character-based LSTM

### Character-based LSTM
* idea from char-rnn (https://github.com/karpathy/char-rnn), some implementation from sherjilozair's tensorflow char-rnn implementation (https://github.com/sherjilozair/char-rnn-tensorflow)
* Korean language model (character based, ... more todo), beam search for k-top result
* Tested on MAC OS X, Windows (in Windows, need to do script works (\*.sh) by yourself)

### Demo for autocompletion
* prerequisites

```
# install requirements (anaconda, tensorflow, etc ...)
cd data
./download.sh # downloading dataset
```

* Train and test text generative model

```
python train.py
python sample.py
```

* Test autocompletion algorithm

```
python main.py
open localhost:~
```
