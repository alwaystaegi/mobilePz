import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from googletrans import Translator
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from flask import Flask
from flask import request

import urllib.parse
import tensorflow_hub as hub

"""
생성된 모델을 이용하여 모델을 구동후 
Flask로 웹서버 구현, 텍스트가 입력되면 결과값을 json으로 내보냄

api 사용방법
255.255.255.255:5000/?text=
255대신 본인의 IPv4주소 혹은 IPv6주소를 대입하고 text attribute에 검사하고자 하는 텍스트를 입력한다.
"""

import statistics

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)


translator = Translator()

mind = tf.keras.models.load_model(
    "./model/mind.h5", compile=False, custom_objects={"KerasLayer": hub_layer}
)
energy = tf.keras.models.load_model(
    "./model/energy.h5", compile=False, custom_objects={"KerasLayer": hub_layer}
)
nature = tf.keras.models.load_model(
    "./model/nature.h5", compile=False, custom_objects={"KerasLayer": hub_layer}
)
tactics = tf.keras.models.load_model(
    "./model/tactics.h5", compile=False, custom_objects={"KerasLayer": hub_layer}
)


def replace_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))  # Load stop words
    pers_types = [
        "INFP",
        "INFJ",
        "INTP",
        "INTJ",
        "ENTP",
        "ENFP",
        "ISTP",
        "ISFP",
        "ENTJ",
        "ISTJ",
        "ENFJ",
        "ISFJ",
        "ESTP",
        "ESFP",
        "ESFJ",
        "ESTJ",
    ]
    pers_types = [p.lower() for p in pers_types]
    try:
        text = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', " ", text)

        text = re.sub("[^0-9a-z]", " ", text)
        text = text.lower()
        text = " ".join(
            [word for word in text.split() if word not in stop_words]
        )  # Remove stop words
        # print(len(sentence))

        for p in pers_types:
            text = re.sub(p, "", text)
        # print(len(sentence))

        text = lemmatizer.lemmatize(text)  # Lemmatize words
    except:
        """"""
    return text


def normalize(res, range, minima):
    normalized_vals = []
    for arr in res:
        normalized_vals.append((arr[0] + abs(minima)) / range)
    return normalized_vals


def float_to_mind(float_results):
    res = {}
    E = []
    I = []
    for num in float_results:
        E.append(1.0 - num)
        I.append(num)

    res["E"] = statistics.fmean(E)

    res["I"] = statistics.fmean(I)

    return res


def float_to_energy(float_results):
    res = {}
    N = []
    S = []
    for num in float_results:
        N.append(1.0 - num)
        S.append(num)

    res["N"] = statistics.fmean(N)

    res["S"] = statistics.fmean(S)

    return res


def float_to_nature(float_results):
    res = {}
    F = []
    T = []
    for num in float_results:
        F.append(1.0 - num)
        T.append(num)

    res["F"] = statistics.fmean(F)

    res["T"] = statistics.fmean(T)

    return res


def float_to_tactics(float_results):
    res = {}
    J = []
    P = []
    for num in float_results:
        J.append(1.0 - num)
        P.append(num)

    res["J"] = statistics.fmean(J)

    res["P"] = statistics.fmean(P)

    return res


def float_to_all(float_results_m, float_results_e, float_results_n, float_results_t):
    res = {}
    E = []
    I = []
    N = []
    S = []
    F = []
    T = []

    J = []
    P = []

    for num in float_results_m:
        E.append(1.0 - num)
        I.append(num)

    for num in float_results_e:
        N.append(1.0 - num)
        S.append(num)

    for num in float_results_n:
        F.append(1.0 - num)
        T.append(num)

    for num in float_results_t:
        J.append(1.0 - num)
        P.append(num)

    res["E"] = statistics.fmean(E)

    res["I"] = statistics.fmean(I)

    res["N"] = statistics.fmean(N)

    res["S"] = statistics.fmean(S)

    res["F"] = statistics.fmean(F)

    res["T"] = statistics.fmean(T)

    res["J"] = statistics.fmean(J)

    res["P"] = statistics.fmean(P)

    return res


def translate_text(text):
    result = ""
    try:
        result = translator.translate(text, dest="en").text

        print(result)
        return result
    except Exception:
        return text


def predict(input_string):
    input_arr = []
    input_arr.append(input_string)

    results_m = mind.predict(input_arr)
    results_e = energy.predict(input_arr)
    results_n = nature.predict(input_arr)
    results_t = tactics.predict(input_arr)

    res = float_to_all(
        normalize(results_m, m_range, m_minima),
        normalize(results_e, e_range, e_minima),
        normalize(results_n, n_range, n_minima),
        normalize(results_t, t_range, t_minima),
    )

    # # Mind
    # results = mind.predict(input_arr)
    # mind_res = (float_to_mind(normalize(results, m_range, m_minima)))

    # # Energy
    # results = energy.predict(input_arr)
    # energy_res = (float_to_energy(normalize(results, e_range, e_minima)))

    # # Nature
    # results = nature.predict(input_arr)
    # nature_res = (float_to_nature(normalize(results, n_range, n_minima)))

    # # Tactics
    # results = tactics.predict(input_arr)
    # tactics_res = (float_to_tactics(normalize(results, t_range, t_minima)))

    return res
    # return mind_res + energy_res + nature_res + tactics_res


mbti_df = pd.read_csv("mbti_1.csv")

mind_names = ["E", "I"]
mind_l = []  # E or I

energy_names = ["N", "S"]
energy_l = []  # N or S

nature_names = ["F", "T"]
nature_l = []  # F or T

tactics_names = ["J", "P"]
tactics_l = []  # J OR P

for t in mbti_df.type:
    mind_l.append(mind_names.index(t[0]))
    energy_l.append(energy_names.index(t[1]))
    nature_l.append(nature_names.index(t[2]))
    tactics_l.append(tactics_names.index(t[3]))

mbti_df["mind"] = mind_l
mbti_df["energy"] = energy_l
mbti_df["nature"] = nature_l
mbti_df["tactics"] = tactics_l


label_mind = mbti_df.mind

mbti_df["posts"] = mbti_df["posts"].apply(replace_text)
# mbti_df['posts']=mbti_df['posts'].apply(translate_text)
# mbti_df.to_csv('mbti_1_kor.csv')

# mbti_df.to_csv('mbti_1_replace.csv')
feature = mbti_df.posts


feature_train, feature_test, labelm_train, labelm_test = train_test_split(
    feature, label_mind, test_size=0.3, random_state=42, stratify=label_mind
)

# print(feature_train)
# Mind
res1 = mind.predict(feature_train)
m_minima = float((min(res1))[0])
m_maxima = float((max(res1))[0])
m_range = m_maxima - m_minima

# Energy
res1 = energy.predict(feature_train)
e_minima = float((min(res1))[0])
e_maxima = float((max(res1))[0])
e_range = e_maxima - e_minima

# Nature
res1 = nature.predict(feature_train)
n_minima = float((min(res1))[0])
n_maxima = float((max(res1))[0])
n_range = n_maxima - n_minima

# Tactics
res1 = tactics.predict(feature_train)
t_minima = float((min(res1))[0])
t_maxima = float((max(res1))[0])
t_range = t_maxima - t_minima


@app.route("/", methods=["GET"])
def hello():
    parameter_dict = request.args.to_dict()

    print(parameter_dict)

    text = parameter_dict["text"]
    text = translate_text(text)

    results = predict(urllib.parse.unquote(text))

    return results


if __name__ == "__main__":
    app.run(host="0.0.0.0")
