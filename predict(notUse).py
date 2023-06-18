import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

mind = tf.keras.models.load_model('./model/mind.h5', compile=False,custom_objects={'KerasLayer':hub_layer})
energy = tf.keras.models.load_model('./model/energy.h5', compile=False,custom_objects={'KerasLayer':hub_layer})
nature = tf.keras.models.load_model('./model/nature.h5', compile=False,custom_objects={'KerasLayer':hub_layer})
tactics = tf.keras.models.load_model('./model/tactics.h5', compile=False,custom_objects={'KerasLayer':hub_layer})




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



def predict(input_string):
    input_arr = []
    input_arr.append(input_string)

  # Mind
    results = mind.predict(input_arr)
    print(results,"mind")
    mind_res = (float_to_mind(normalize(results, m_range, m_minima)))[0]

  # Energy
    results = energy.predict(input_arr)
    print(results,"Energy")

    energy_res = (float_to_energy(normalize(results, e_range, e_minima)))[0]
  
  # Nature
    results = nature.predict(input_arr)
    print(results,"Nature")
    nature_res = (float_to_nature(normalize(results, n_range, n_minima)))[0]

  # Tactics
    results = tactics.predict(input_arr)
    print(results,"Tactics")
    
    tactics_res = (float_to_tactics(normalize(results, t_range, t_minima)))[0]

    return mind_res + energy_res + nature_res + tactics_res






print(1)


x_test= ['hello my name is test']


# Mind
res1 = mind.predict(x_test)

m_minima = float((min(res1))[0])
m_maxima = float((max(res1))[0])
m_range = m_maxima-m_minima

# Energy
res1 = energy.predict(x_test)
e_minima = float((min(res1))[0])
e_maxima = float((max(res1))[0])
e_range = e_maxima-e_minima

# Nature
res1 = nature.predict(x_test)
n_minima = float((min(res1))[0])
n_maxima = float((max(res1))[0])
n_range = n_maxima-n_minima

# Tactics
res1 = tactics.predict(x_test)
t_minima = float((min(res1))[0])
t_maxima = float((max(res1))[0])
t_range = t_maxima-t_minima



# 결과리스트,최소-최대,최소
# 계산식 res가 3중배열로 되어있음 [[[]],[[[]]],[[]]]

mind_res = (float_to_mind(normalize(res1, m_range, m_minima)))[0]


result=predict('hello im good')

print(result)