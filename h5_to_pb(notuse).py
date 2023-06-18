from tensorflow import keras
import tensorflow_hub as hub
import tensorflow as tf



energy = keras.models.load_model('energy.h5', compile=False)
mind = keras.models.load_model('mind.h5', compile=False)
nature = keras.models.load_model('nature.h5', compile=False)
tactics = keras.models.load_model('tactics.h5', compile=False)



export_path = './pb'
energy.save('./energys', save_format='tf')

mind.save('./minds', save_format='tf')
nature.save('./natures', save_format='tf')
tactics.save('./tacticss', save_format='tf')
