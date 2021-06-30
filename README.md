# fastread

How readable is your text?

This repo is to build a ML service which predicts readability of any text by combining Fastai API and NLP models from Kaggle CommonLit competition.


### How to run this app on your local machine

Before going forward, install git-lfs from https://git-lfs.github.com/ to download a big model file from git.

`git clone https://github.com/faceon/fastread.git`
`cd fastread`
`conda env create -f environment.yml`
`conda activate fastread`
`streamlit run streamlit_app.py`