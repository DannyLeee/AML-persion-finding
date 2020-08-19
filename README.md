# Introduction
This is a part of competition using news to find possible (or used to) money laundering person. Finding these people can help Anti-money laundering (AML) in future. This repository only focus on predict BIO tagging.

# Model Structure
* Input the news through language model (here use [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) pretrained on training set).
* Use the CLS token embedding predict if the news have the target person.[1]
    * If have, use each token embedding output by LM to predict BIO tagging.
![](https://i.imgur.com/UnGfKFe.jpg)
* Usually model output some spans not only some names.
    * So, use the [ckiptrigger](https://github.com/ckiplab/ckiptagger) to find person name in these spans.[2]
    * But ckiptrigger sometime get wrong names, use QA-base model check ckip output's name is or not the target person.[3]

# Acknowledgments
Thank [@Leo Lin](https://github.com/CoyoteLeo) build web crawler and arrange training set, inference ckiptrigger[2] and integrate team members' code into API; [@HongYun0901
](https://github.com/HongYun0901) build the QA-base model for the final model[3]; [@Mouthhan](https://github.com/Mouthhan) and [@Edward Wu](https://github.com/Marzear) build the binary classifier to predict have or not target person[1].

# Postscript
Finally, our team found only use QA-base input news and ckiptrigger output's can have better performance. So this repository (BIO model) nerver use in latter half of the competition. We think it could because the training data are too few to train a BIO tagger although it has convergence.
* The final competition model inference is [here](https://github.com/CoyoteLeo/T-Brain-2020-Summer-NLP)

# Dataset, Pretrained Model and Colab
The dataset, pretrain model, final model and colab notebook of this repository can be download from [here](https://drive.google.com/drive/folders/1r5kAb6NY0LXU2ldLjWm7V4i8p74xLani?usp=sharing)

---

**Notice: This repository python file and notebook can not totally execute because some models(QA-base) are loss.**