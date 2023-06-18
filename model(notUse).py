import pandas as pd

from text import *

from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
from tqdm import tqdm



mbti_df = pd.read_csv("mbti_1.csv")

mind_names = ["E" , "I"]
mind = []  # E or I

energy_names = ["N",  "S"]
energy = [] # N or S

nature_names = ["F", "T"]
nature = [] # F or T

tactics_names = ["J", "P"]

tactics = [] # J OR P

for t in mbti_df.type:
    mind.append(mind_names.index(t[0]))
    energy.append(energy_names.index(t[1]))
    nature.append(nature_names.index(t[2]))
    tactics.append(tactics_names.index(t[3]))

mbti_df['mind'] = mind
mbti_df['energy'] = energy
mbti_df['nature'] = nature
mbti_df['tactics'] = tactics

mbti_df['posts']=mbti_df['posts'].apply(replace_text)
mbti_df['posts']=mbti_df['posts'].apply(translate_text)
mbti_df.to_csv('mbti_1_kor.csv')

d = {'INTJ':'You are thoughtful, rational, quick-witted and independent. However, sometimes you are known to be overly critical, and have a combative side to yourself.', 'INTP':'You are unique, creative, inventive and imaginative. However, sometimes you are known to be a bit insensitive and impatient with others.', 'ENTJ':'You are determined, charismatic, confident and authoritative. However, sometimes you are known to  be intolerant of other people\'s weaknesses, and slightly arrogant.', 'ENTP':'You are audacious, bold, playful and rebellious. However, sometimes you can find it difficult to focus, and dislike talking about practical matters.', 'INFJ':'You are creative, insightful, passionate and have strong morals. However, sometimes you are a bit of a perfectionist, and find it reluctant to open up to other people.','INFP':'You are empathetic, generous, creative and passionate. However, sometimes your goals are a bit unrealistic and you tend to lack focus sometimes.', 'ENFJ':'You are passionate, reliable, charismatic and very receptive. However, sometimes you can be overly empathetic and condescending toward other people.', 'ENFP' : 'You are enthusiastic, festive, good-natured and excellent at communicating. However, you sometimes focus on being a people pleaser and disorganized.', 'ISTJ': 'You are very responsible, strong-willed, calm and enforce order. However, you are known to be stubborn and are somewhat judgemental sometimes', 'ISFJ': 'You are reliable, observant, enthusiastic and supportive. However, you are known to be overly humble and tend to take things personally', 'ESTJ' : 'You are dedicated, strong-willed, loyal and reliable. However, you find it difficult to relax, or share what you\'re feeling with other people.', 'ESFJ': 'You are very loyal, sensitive to other people\'s feelings, and have strong practical skills. However, you are sometimes worried about your social status and tend to be vulnerable to criticism.', 'ISTP': 'You are spotaneous, rational, optimistic and know how to prioritize things. However, you are known to be stubborn and get bored very easily.',  'ISFP' : 'You are charming, imaginative, passionate and sensitive to others. However, you are fiercely independent and get stressed out pretty easily.', 'ESTP': 'You are perceptive, direct, bold and rational. However, you tend to be defiant and may sometimes miss the bigger picture in favor of smaller victories.', 'ESFP' : 'You are observant, practical, have excellent people skills and are fond of showmanship. However, you are very sensitive and sometimes avoid conflict entirely.'}


label_mind = mbti_df.mind
label_energy = mbti_df.energy
label_nature = mbti_df.nature
label_tactics = mbti_df.tactics
feature = mbti_df.posts

feature_train, feature_test, labelm_train, labelm_test = train_test_split (feature, label_mind, test_size =.3 , random_state= 42, stratify= label_mind)
feature_train, feature_test, labele_train, labele_test = train_test_split (feature, label_energy, test_size =.3 , random_state= 42, stratify= label_energy)
feature_train, feature_test, labeln_train, labeln_test = train_test_split (feature, label_nature, test_size =.3 , random_state= 42, stratify= label_nature)
feature_train, feature_test, labelt_train, labelt_test = train_test_split (feature, label_tactics, test_size =.3 , random_state= 42, stratify= label_nature)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

def createModel():
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()
    model.compile(optimizer='adam',
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
    return model

def normalize(res, range, minima):
    normalized_vals = []
    for arr in res:
        normalized_vals.append((arr[0] + abs(minima))/range)
    return normalized_vals

def float_to_mind(float_results):
    res = []
    for num in float_results:
        if(num < 0.5):
            res.append("E")
        else:
            res.append("I")
    return res


def float_to_energy(float_results):
    res = []
    for num in float_results:
        if(num < 0.5):
            res.append("N")
        else:
            res.append("S")
    return res


def float_to_nature(float_results):
    res = []
    for num in float_results:
        if(num < 0.5):
            res.append("F")
        else:
            res.append("T")
    return res   

def float_to_tactics(float_results):
    res = []
    for num in float_results:
        if(num < 0.5):
            res.append("J")
        else:
            res.append("P")
    return res


feature_val = feature_train[3036:]
partial_feature_train = feature_train[:3036]

m_val = labelm_train[3036:]
partial_m_train = labelm_train[:3036]

e_val = labele_train[3036:]
partial_e_train = labele_train[:3036]

n_val = labeln_train[3036:]
partial_n_train = labeln_train[:3036]

t_val = labelt_train[3036:]
partial_t_train = labelt_train[:3036]



mind = createModel()
history = mind.fit(partial_feature_train,
                    partial_m_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(feature_val, m_val),
                    verbose=0)

# tfjs.converters.save_keras_model(mind, "mind")

energy = createModel()
history = energy.fit(partial_feature_train,
                    partial_e_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(feature_val, e_val),
                    verbose=0)
# tfjs.converters.save_keras_model(energy, "energy")

nature = createModel()
history = nature.fit(partial_feature_train,
                    partial_n_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(feature_val, n_val),
                    verbose=0)
# tfjs.converters.save_keras_model(model, "nature")


tactics = createModel()
history = tactics.fit(partial_feature_train,
                    partial_t_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(feature_val, t_val),
                    verbose=0)
#tfjs.converters.save_keras_model(model, "tactics")



# Mind
res1 = mind.predict(feature_train)
m_minima = float((min(res1))[0])
m_maxima = float((max(res1))[0])
m_range = m_maxima-m_minima

# Energy
res1 = energy.predict(feature_train)
e_minima = float((min(res1))[0])
e_maxima = float((max(res1))[0])
e_range = e_maxima-e_minima

# Nature
res1 = nature.predict(feature_train)
n_minima = float((min(res1))[0])
n_maxima = float((max(res1))[0])
n_range = n_maxima-n_minima

# Tactics
res1 = tactics.predict(feature_train)
t_minima = float((min(res1))[0])
t_maxima = float((max(res1))[0])
t_range = t_maxima-t_minima


def predict(input_string):
    input_arr = []
    input_arr.append(input_string)

    # Mind
    results = mind.predict(input_arr)
    mind_res = (float_to_mind(normalize(results, m_range, m_minima)))[0]

    # Energy
    results = energy.predict(input_arr)
    energy_res = (float_to_energy(normalize(results, e_range, e_minima)))[0]
  
    # Nature
    results = nature.predict(input_arr)
    nature_res = (float_to_nature(normalize(results, n_range, n_minima)))[0]

    # Tactics
    results = tactics.predict(input_arr)
    tactics_res = (float_to_tactics(normalize(results, t_range, t_minima)))[0]

    return mind_res + energy_res + nature_res + tactics_res



results = predict(input())
print(results)
print(d[results])

mind.save('mind.h5')
energy.save('energy.h5')
nature.save('nature.h5')
tactics.save('tactics.h5')

mind_converter=tf.lite.TFLiteConverter.from_keras_model(mind)
energy_converter=tf.lite.TFLiteConverter.from_keras_model(energy)
nature_converter=tf.lite.TFLiteConverter.from_keras_model(nature)
tactics_converter=tf.lite.TFLiteConverter.from_keras_model(tactics)



mind_model=mind_converter.convert()
energy_model=energy_converter.convert()
nature_model=nature_converter.convert()
tactics_model=tactics_converter.convert()



with open('mind_model.tflite', 'wb') as f:
    f.write(mind_model)
with open('energy_model.tflite', 'wb') as f:
    f.write(energy_model)
with open('nature_model.tflite', 'wb') as f:
    f.write(nature_model)
with open('tactics_model.tflite', 'wb') as f:
    f.write(tactics_model)


