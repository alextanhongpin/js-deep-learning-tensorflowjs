(() => {
  window.addEventListener("load", onload, false);

  async function onload() {
    const mnist = new MNISTPredictor();

    const $clear = document.getElementById("clear");
    const $predict = document.getElementById("predict");
    const $output = document.getElementById("output");

    const canvas = document.getElementById("canvas");
    canvas.width = 200;
    canvas.height = 200;

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let isMousedown = false;
    canvas.addEventListener(
      "mousedown",
      (evt) => {
        isMousedown = true;
        ctx.beginPath();
        ctx.moveTo(evt.pageX, evt.pageY);
        ctx.lineWidth = 10;
        ctx.stroke();
      },
      false
    );

    canvas.addEventListener(
      "mousemove",
      (evt) => {
        if (!isMousedown) return;

        ctx.lineTo(evt.pageX, evt.pageY);
        ctx.stroke();
      },
      false
    );

    canvas.addEventListener(
      "mouseup",
      (evt) => {
        isMousedown = false;
        ctx.closePath();
      },
      false
    );

    // This is where the prediction is made.
    $predict.addEventListener(
      "click",
      async () => {
        const number = await mnist.predict(canvas);
        $output.innerHTML = `Predicted: ${number}`;
      },
      false
    );

    $clear.addEventListener(
      "click",
      () => {
        ctx.reset();
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      },
      false
    );
  }

  // MNISTPredictor predicts a number 0-9 from a given image.
  class MNISTPredictor {
    constructor(width = 28, height = 28) {
      this.width = width;
      this.height = height;
      this.model = null;
    }

    // init loads the pre-trained MNIST model.
    async init() {
      if (this.model) return;
      this.model = await tf.loadLayersModel("/model/model.json");
    }

    // preprocessCanvas converts a canvas image into the input 4d tensors
    // for tensorflow to make predictions.
    preprocessCanvas(canvasElement) {
      const channel = 1;
      // Load the image as tensors with only 1 channel (grayscale).
      let x = tf.browser.fromPixels(canvasElement, channel);

      // Tensorflow operates in single-precision.
      x = x.asType("float32");

      // Resize the image size to fit the input.
      x = tf.image.resizeBilinear(x, [this.height, this.width]);

      // The train images have black background and white text, but our image
      // has white background and black text.
      // We inverse the pixel by substracting 255 from it.
      x = tf.sub(255, x);

      // Normalize the pixels so that all values stays between 0-1 instead of
      // 0-255.
      x = x.div(255);

      // For debugging (this is actually async).
      //tf.browser.toPixels(x, canvasElement);

      // The tensor created by tf.browser.fromPixels does not include a batch
      // dimension.
      // Converts the 3d-tensors to 4d-tensors.
      x = x.expandDims();

      return x;
    }

    // predict predicts a number 0-9 from the canvas image.
    async predict(canvas) {
      if (!this.model) await this.init();

      const x = this.preprocessCanvas(canvas);
      const output = this.model.predict(x);
      const axis = 1;
      const predictions = Array.from(output.argMax(axis).dataSync());
      return predictions[0];
    }
  }
})();
