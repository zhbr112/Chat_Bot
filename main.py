import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel, BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")

tokenizer.save_pretrained("./rubert-tokenizer")
model.save_pretrained("./rubert-model")

bert_tokenizer = AutoTokenizer.from_pretrained("./rubert-tokenizer")
bert_model = TFAutoModel.from_pretrained("./rubert-model")

def preprocess_text(text: list, model, tokenizer):
    return model(**tokenizer(text, return_tensors='tf', padding=True, truncation=True))['last_hidden_state'][:, 0, :].numpy()

df = pd.read_csv('/content/DataSet.csv', sep=';')

df

dataset = []
for index, row in df.iterrows():
    embedding_question = preprocess_text([row['Question']], bert_model, bert_tokenizer)
    embedding_answer = preprocess_text([row['Answer']], bert_model, bert_tokenizer)
    dataset.append([embedding_question[0], embedding_answer[0]])

dataset = np.array(dataset)

dataset

X, Y = [], []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
        X.append(np.concatenate([dataset[i, 0, :], dataset[j, 1, :]], axis=0))
        Y.append(1 if i == j else 0)

X = np.array(X)
Y = np.array(Y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1],)))
model.add(tf.keras.layers.Dense(100, activation='selu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])
model.fit(X, Y, epochs=2000, class_weight={0: 1, 1: np.sqrt(Y.shape[0])-1})

model.save("./rubert-model")

def get_answer(question):
    embedding_question = preprocess_text([question], bert_model, bert_tokenizer)[0]

    p = []
    for i in range(dataset.shape[0]):
        embedding_answer = dataset[i, 1]
        combined_embedding = np.concatenate([embedding_question, embedding_answer])
        prediction = model.predict(np.expand_dims(combined_embedding, axis=0), verbose=False)[0, 0]
        p.append([i, prediction])

    p = np.array(p)
    ans = np.argmax(p[:, 1])

    return df["Answer"][ans]
def chat_bot():
  print("Введите вопрос или выход")
  while True:
        question = input("Введите вопрос: ")
        if question.lower() == "выход":
            break

        answer = get_answer(question)
        print("Ответ:", answer)

chat_bot()