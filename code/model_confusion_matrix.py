from data_process import *
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


path_tfrecords = '../tfrecords/train_nocontainmotion.tfrecords'
predict_model = tf.keras.models.load_model('../model/posture_classify_highacc.h5', compile=True) #  注意这儿得compile需要设置为true，如果你不设置你需要多一步compile的过程。
test_dataset = gen_valdata_batch(path_tfrecords, cfg.batch_size)
print('-----------test--------')
test_loss, test_acc = predict_model.evaluate(test_dataset, verbose=1)
print('测试准确率：', test_acc)
print('测试损失', test_loss)

y_predict = predict_model.predict(test_dataset, batch_size=cfg.batch_size, verbose=1)
y_predict = np.argmax(y_predict, axis=3)
y_predict = y_predict.flatten().tolist()
print(y_predict)
y_true = get_truelabel(path_tfrecords)
print(y_true)

confusionmatrix = confusion_matrix(y_true, y_predict) #行的标签代表真实值，列的标签代表预测值
print(confusionmatrix)
confusionmatrix_percent = []
for matrix in confusionmatrix:
    matrix_sum = np.sum(confusionmatrix, axis=0)
    # print(matrix_sum)
    matrix = matrix / matrix_sum
    # print(matrix)
    confusionmatrix_percent.append(matrix)
# print(confusionmatrix_percent)

# x_ticks = ["supine", "right latericumbent", "left latericumbent", "prone"]
# y_ticks = ["supine", "right latericumbent", "left latericumbent", "prone"]  # 自定义横纵轴

x_ticks = ["supine", "right", "left", "prone"]
y_ticks = ["supine", "right", "left", "prone"]  # 自定义横纵轴
ax = sns.heatmap(confusionmatrix_percent, xticklabels=x_ticks, yticklabels=y_ticks, annot=True, cbar=None, cmap='Blues', fmt='.3f')
# ax = sns.heatmap(confusionmatrix, xticklabels=x_ticks, yticklabels=y_ticks, annot=True, cbar=None, cmap='Blues', fmt='g')
ax.set_title('Confusion Matrix')  # 图标题
ax.set_xlabel('True Label')  # x轴标题
ax.set_ylabel('Predict Label')
plt.show()
figure = ax.get_figure()
figure.savefig('../result_img/sns_heatmap.jpg')  # 保存图片