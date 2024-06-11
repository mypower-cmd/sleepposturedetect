from data_process import *


predict_model = tf.keras.models.load_model("model/posture_classify.h5", compile=True) #  注意这儿得compile需要设置为true，如果你不设置你需要多一步compile的过程。
test_dataset = gen_data_batch(cfg.test_dataset_path,
                              cfg.batch_size,
                              is_training=False)
y_predict = predict_model.predict(test_dataset)
print(y_predict)