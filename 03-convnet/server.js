import tf from "@tensorflow/tfjs-node";
import data from "./data.js";
const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;

async function main() {
  await data.loadData();
  console.log(data);

  const model = createConvModel();
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: "rmsprop",
    metrics: ["accuracy"],
  });

  const trainData = data.getTrainData();
  const testData = data.getTestData();
  const epochs = 1;

  const batchSize = 320;
  const validationSplit = 0.15;
  await model.fit(trainData.images, trainData.labels, {
    batchSize,
    validationSplit,
    epochs,
    onBatchEnd: async (batch, logs) => {},
    onEpochEnd: async (epoch, logs) => {
      console.log({ epoch, logs });
    },
  });

  const result = model.evaluate(testData.images, testData.labels);
  console.log(result);

  await model.save(`file://model`);
}

function createConvModel() {
  const model = tf.sequential();

  // First layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1],
      kernelSize: 3,
      filters: 32,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: "relu",
    })
  );
  // Pooling after convolution.
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  );

  // Second layer.
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 64,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.conv2d({
      kernelSize: 3,
      filters: 64,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  );

  // Flatten tensors to prepare for dense layers.
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.25 })); // To reduce overfitting.
  model.add(
    tf.layers.dense({
      units: 512,
      activation: "relu",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.5 })); // To reduce overfitting.

  // Use softmax for multiclass classification problem.
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "softmax",
    })
  );
  model.summary();
  return model;
}

main().catch(console.error);
