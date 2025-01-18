import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
import time
from optoolkit import Model, DatasetLoader, Trainer


if __name__ == "__main__":
    # Load dataset
    start_time = time.time()
    dataset = DatasetLoader(["Message"], "Message")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('readerbench/RoBERT-small')
    
    # Encode dataset
    encoded_dataset = dataset.encode_dataset(tokenizer)

    # K-Fold Cross Validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    all_metrics = []

    # Prepare for K-Fold Cross Validation
    all_labels = np.array(encoded_dataset['data'])  # Assuming encoded_dataset['labels'] holds the multi-labels

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_labels)):
        print(f'Fold {fold + 1}/{k_folds}')
        
        # Create DataLoader for current fold
        train_subset = encoded_dataset['data'].select(train_idx)
        val_subset = encoded_dataset['data'].select(val_idx)

        # Initialize model for this fold
        model = Model(model_name='./automod-model/model_versions/v1', 
                                       num_labels=len(dataset.labels), 
                                       id2label=dataset.id2label, 
                                       label2id=dataset.label2id)

        # Initialize trainer for this fold
        trainer = Trainer(model=model.model, 
                                  tokenizer=tokenizer, 
                                  train_dataset=train_subset, 
                                  eval_dataset=val_subset, 
                                  output_dir=f'./automod-model/model_versions/v2-fold-{fold + 1}',
                                  batch_size=16,
                                  epochs=8
                                  )

        # Train and evaluate model for the current fold
        trainer.train()
        metrics = trainer.evaluate()
        all_metrics.append(metrics)

        # Save model for the current fold
        trainer.save_model()

    # Calculate average metrics across all folds
    avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0]}
    print(f'Average Metrics: {avg_metrics}')

    # Example inference
    text = "mai taci in rasa ma-tii"
    logits = model.predict(text, tokenizer)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    predicted_labels = [dataset.id2label[idx] for idx, label in enumerate(predictions) if label]
    
    print(predicted_labels)
    print(predictions)
    print(probs)

    end_time = time.time()
    print(f'Execution time: {(end_time - start_time):.2f} seconds.')
