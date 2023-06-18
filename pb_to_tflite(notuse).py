import tensorflow as tf


path=['energy','mind','nature','tactics']
# path=['energy']


for i in path:
    converter = tf.lite.TFLiteConverter.from_saved_model('./'+i)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open('./tf/'+i+'.tflite', 'wb').write(tflite_model)


