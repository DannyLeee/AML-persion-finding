# Introduction
This is a part of competition using news to find possible (or used to) money laundering person. Finding these people can help Anti-money laundering (AML) in future. This repositroies only focus on predict BIO tagging.

# Model Structure
* Input the news through language model (here use [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) pretrained on training set).
* Use the CLS token embedding predict if the news have the target person.[1]
    * If have, use each token embedding output by LM to predict BIO tagging. 
* Usually model output some spans not only some names.
    * So, use the [ckiptrigger](https://github.com/ckiplab/ckiptagger) to find person name in these spans.
    * But ckiptrigger sometime get wrong names, use QA-base model check ckip output's name is or not the target person.[2]

# Acknowledgments
Thank [@Leo Lin](https://github.com/CoyoteLeo) build web crawler and arrange training set, inference ckiptrigger and integrate team members' code into API; [@HongYun0901
](https://github.com/HongYun0901) build the QA-base model for the final model[2]; [@Mouthhan](https://github.com/Mouthhan) and [@Edward Wu](https://github.com/Marzear) build the binary classifier to predict have or not target person[1].

# postscript
Final our team found only use QA-base input news and ckiptrigger output's can have better performance. So this model nerver use in latter half of the competition. We think it could because the training data are too few to train a BIO tagger although it has convergence.
* The final model inference is [here](https://github.com/CoyoteLeo/T-Brain-2020-Summer-NLP)

# Dataset, Pretrained Model and Colab
The dataset, pretrain model and colab notebook can be download from [here](https://drive.google.com/drive/folders/1r5kAb6NY0LXU2ldLjWm7V4i8p74xLani?usp=sharing)