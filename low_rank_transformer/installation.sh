conda create -n transformer python=3.6
source activate transformer

#conda install pytorch torchvision torchtext cudatoolkit=10.1 -c pytorch
conda install pytorch==1.6.0 torchvision==0.7.0 torchtext=0.4.0 cudatoolkit=10.1 -c pytorch
conda install -c anaconda spacy=2.3.5
conda install -c anaconda dill

python -m spacy download en
python -m spacy download de

#git clone https://github.com/hwang595/dist-transformer.git
#cd dist-transformer

python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl