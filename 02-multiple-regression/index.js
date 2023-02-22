import { BostonDataset } from "./data.js";

async function load() {
  const dataset = new BostonDataset();
  const [[trainX, trainY], [testX, testY]] = await dataset.load();
  const tensors = {
    trainX: normalize(tf.tensor2d(trainX)),
    trainY: tf.tensor2d(trainY),
    testX: normalize(tf.tensor2d(testX)),
    testY: tf.tensor2d(testY),
  };

  computeBaseline(tensors);

  const model = tf.sequential();
  /*
  model.add(
    tf.layers.dense({
      inputShape: [dataset.numFeatures],
      units: 1, // We expect output price.
    })
  );
*/
  model.add(
    tf.layers.dense({
      inputShape: [dataset.numFeatures],
      units: 50, // We expect output price.
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(
    tf.layers.dense({
      units: 1,
    })
  );
  model.summary();
  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: "meanSquaredError",
  });

  const NUM_EPOCHS = 20;
  const BATCH_SIZE = 32;
  console.log(tensors.trainX);
  await model.fit(tensors.trainX, tensors.trainY, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed. loss: ${
            logs.loss
          } val_loss: ${logs.val_loss}`
        );
      },
    },
  });

  model
    .evaluate(tensors.testX, tensors.testY, { batchSize: BATCH_SIZE })
    .print();
}

// computeBaseline computes the mean baseline mean squared error without
// training the model.
function computeBaseline({ trainY, testY }) {
  const avgPrice = tf.mean(trainY);
  console.log(`Average price: ${avgPrice.dataSync()[0]}`);

  const baseline = tf.mean(tf.pow(tf.sub(testY, avgPrice), 2));
  console.log(`Baseline loss: ${baseline.dataSync()[0]}`);
}

// normalize normalizes the data by subtracting the mean and dividing it by the
// standard deviation.
function normalize(data) {
  const mean = data.mean(0);
  const variance = data.sub(mean).square().mean(0);
  const std = variance.sqrt();

  return data.sub(mean).div(std);
}

window.addEventListener("load", load);
