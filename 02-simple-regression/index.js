// Example: Predicting download duration with tensorflow.js.

(() => {
  const start = window.performance.now();
  const $output = document.getElementById("output");
  $output.innerHTML = "Training...";

  // Prepare test and train dataset.
  const trainData = {
    sizeMB: [
      0.08, 9.0, 0.001, 0.1, 8.0, 5.0, 0.1, 6.0, 0.05, 0.5, 0.002, 2.0, 0.005,
      10.0, 0.01, 7.0, 6.0, 5.0, 1.0, 1.0,
    ],
    timeSec: [
      0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116,
      0.07, 0.289, 0.076, 0.744, 0.083, 0.56, 0.48, 0.399, 0.153, 0.149,
    ],
  };

  const testData = {
    sizeMB: [
      5.0, 0.2, 0.001, 9.0, 0.002, 0.02, 0.008, 4.0, 0.001, 1.0, 0.005, 0.08,
      0.8, 0.2, 0.05, 7.0, 0.005, 0.002, 8.0, 0.008,
    ],
    timeSec: [
      0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.07, 0.375, 0.058, 0.136,
      0.052, 0.063, 0.183, 0.087, 0.066, 0.558, 0.066, 0.068, 0.61, 0.057,
    ],
  };

  // Converting data into tensors.
  const trainTensors = {
    // [20, 1] refers to the tensors "shape".
    // We have 20 samples where each sample is 1 number.
    sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(trainData.timeSec, [20, 1]),
  };

  const testTensors = {
    sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(testData.timeSec, [20, 1]),
  };

  // Defining a simple model.
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [1], // The input should be 1D tensor.
      units: 1, // The output should be one number.
    })
  );
  model.compile({
    optimizer: "sgd",
    loss: "meanAbsoluteError",
  });

  // Do not block the main UI thread.
  (async function () {
    await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {
      epochs: 100,
    });

    // Print out the mean abolute error.
    // A result of 0.05 means the estimates are within 0.05s.
    model.evaluate(testTensors.sizeMB, testTensors.timeSec).print();
    // Output: 0.05498236045241356

    $output.innerHTML = `Done. Took ${window.performance.now() - start}s.`;

    // Make prediction.
    const smallFileMB = 1;
    const largeFileMB = 100;
    const hugeFileMB = 10_000;

    model
      .predict(tf.tensor2d([[smallFileMB], [largeFileMB], [hugeFileMB]]))
      .print();
    /* Output:
    [[0.1736424  ],
     [9.3452492  ],
     [926.5058594]]

    In the train data, 1MB takes about 0.15s (the last two items). 
    The predicted value is 0.173s.
      */
  })();
})();
