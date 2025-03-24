import models
import json
import tensorflow as tf

university = 'UFPR04'
random = True
dataset_json = "/home/vhns/ml/batch_results/2024-06-25T18:15-03:00/dataset_UFPR04.json"
dataset = json.load(open(dataset_json))
num_classes = 2

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


for j in range(1,4):
    for i in range(1,7):
        model = None
        model_path = None
        trained_weights = None
        evaluation_loss = None
        evaluation_accuracy = None
        model_path = f'/home/vhns/ml/batch_results/2024-06-25T18:15-03:00/threecvnnclassifier/{university}_{j}_{i}'
        trained_weights = f'{model_path}/best.weights.h5'
        model = models.ThreeCvnnClassifier(num_classes=num_classes, random=random, dataset=dataset, model_path=model_path, epochs=None, pretrained_weights=None, logger=None)
        model.load_weights(trained_weights)
        model.compile(optimizer='Adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                      metrics=['accuracy'])
        evaluation_loss, evaluation_accuracy = model.evaluate(model.validation_dataset.batch(32))
        #print(model.predict(model.test_dataset.batch(1)))
        evaluation = f'Loss: {evaluation_loss} Accuracy: {evaluation_accuracy}'
        with open(f'{model_path}/full_evaluation_result_vs_validation.txt', 'a') as f:
            print(evaluation, file=f)
