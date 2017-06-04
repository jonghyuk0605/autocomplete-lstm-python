wget https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-v1/korean-english-park.dev.tar.gz
wget https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-v1/korean-english-park.test.tar.gz
wget https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-v1/korean-english-park.train.tar.gz
gunzip *.gz
tar xvf korean-english-park.dev.tar
tar xvf korean-english-park.test.tar
tar xvf korean-english-park.train.tar
rm *.tar
rm *.en
