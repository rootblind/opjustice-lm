
# OPJUSTICE Language Model (project on going)

This project aims to become a collection of language models that will bring any chatbot to a new level of engagement and moderation.

Present models:
- automod-model: Multi label text classification model that can recognize up to 8 categories

Future models:
- chatbot-model: Will give life to
- image-model: In completion to automod-model, image-model will make sure to flag messages that contain NSFW media

opjustice-lm comes as an extension of my other project, justice-bot, through its API.

Visit my discord bot project [here](https://github.com/rootblind/justice-bot).
 

 This project is ON GOING, you can raise **issues** if you wish to contribute!

# Automod-Model

The following sections will give info about the automod-model.
## Labels

You can find an example [here](https://github.com/rootblind/opjustice-lm/blob/main/automod-model/example_dataset.csv).

- OK: There is nothing wrong with this message
- Insult: Bad usage of words against another person
- Violence: Showing or encouraging violence
- Sexual: Sexual content
- Hateful: Usage of slurs or hating a group of people
- Flirt: Showing romantic intent towards another person
- Spam
- Aggro: Content that might start an argument with someone else

## Model version
No model version is yet public, will update this section with the huggingface repository in the future.

## Dataset

The model is trained mainly on Romanian language and a very small sample of English.

The dataset consists of discord messages sent on my discord server (League of Legends Romania) using a discord bot to scrape the text channels for messages (you can check that [here](https://github.com/rootblind/dataminer-bot)).

The labeling is done both manually and self thought through the help of another model [readerbench/ro-offense](https://huggingface.co/readerbench/ro-offense).

The dataset that the model currently uses can be found on huggingface [here](https://huggingface.co/datasets/rootblind/opjustice-dataset).

## Scripts

In the `python_tools` directory there are scripts that I used to work with the data.

Training scripts:
- prepare-dataset: Uses the model to label an unlabeled dataset, it can optionally append the newly labeled dataset to an existing one.
- append_csv: appends a dataset to another, I use this functionality by running this script instead of inside prepare-dataset because that way I can check and correct the results
- load_model: Loading the model and sending text input to test out the outputs
- model: The training script

## Training Loop

Having an initial model and dataset, I would collect unlabeled data, then use prepare-dataset on it, correct the labeling, use append_csv, run model.py on the new data and use load_model to test the new model that resulted.

Then I would update inside the scripts which model version to use for the next training loop.


## Demo / Tutorial
This section will be updated at a later date when the project is mature enough.

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
- Code example [Fine-tuning BERT (and friends) for multi-label text classification](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=mEkAQleMMT0k)


## Author

- [@rootblind](https://www.github.com/rootblind)


## License

[GPL v3](https://github.com/rootblind/justice-bot/blob/main/LICENSE)

