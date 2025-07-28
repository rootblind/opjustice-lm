
# OPJUSTICE Language Models

This project aims to become a collection of language models that will bring any chatbot to a new level of engagement and moderation.

Present models:
- automod: Multi label text classification model that can recognize up to 8 categories
- chatbot: Transformer GPT2-like model


opjustice-lm comes as an extension of my other project, justice-bot, through its API.

Visit my discord bot project [here](https://github.com/rootblind/justice-bot).
 
# Automod Model

The following sections will give info about the automod.
## Labels

You can find an example [here](https://github.com/rootblind/opjustice-lm/blob/main/automod-model/example_dataset.csv).

- OK: There is nothing wrong with this message
- Aggro: Content that might start an argument with someone else through insults or other toxic behaviors
- Violence: Graphical descriptions of or encouraging violence
- Sexual: Sexual content
- Hateful: Usage of slurs or hating a group of people

## Model version
- [Hugging Face](https://huggingface.co/rootblind/opjustice-lm/tree/main) repository

## Dataset

- [Hugging Face](https://huggingface.co/datasets/rootblind/opjustice-dataset)

The model is trained mainly on Romanian language and very small samples of English.

The dataset consists of discord messages sent on my discord server (League of Legends Romania) using a discord bot to scrape the text channels for messages (you can check that [here](https://github.com/rootblind/dataminer-bot)).

The labeling is done both manually and automated through regex based logic.


## Training Loops

Check out [this](https://github.com/rootblind/opjustice-lm/blob/main/automod-model/docs/training_loops.md) readme used to note observations about the model after each training loop.


# About project section

## How to Use

Clone the project

```bash
  git clone https://github.com/rootblind/opjustice-lm
```
Go to the project directory

```bash
  cd opjustice-lm
```

Install dependencies

```bash
  pip install -r requirements.txt
  #make sure to be in the project folder
```

Use Python to run the model on local host

Make sure to replace the <> with your use case.
```bash
  python ./<the_model>/API/model_fastapi.py
```

In case you're using a virtual environment, make sure to run the `activate.bat` script first in your command line. It can be found in `/venv/Scripts/`.


## Get the latest Python version from here:
[Click](https://www.python.org/)

## Technologies used
 - [PyTorch](https://pytorch.org/)
 - [Transformers](https://huggingface.co/docs/transformers/index)
 - [scikit-learn](https://scikit-learn.org/stable/)
## Credits
- Code example for the automod model: [Fine-tuning BERT (and friends) for multi-label text classification](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=mEkAQleMMT0k)

- Video course used for the chatbot model: [Train Your Own LLM – Tutorial](https://youtu.be/9Ge0sMm65jo)

# Related work:
- [OPJUSTICE - A Multi-label Text Classification Model](https://repository.ifipaiai.org/2025/abstr/25dc05431.html) paper submitted at AIAI2025

## Author

- [@rootblind](https://www.github.com/rootblind)


## License

[GPL v3](https://github.com/rootblind/justice-bot/blob/main/LICENSE)

