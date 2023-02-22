window.addEventListener("load", onload, false);
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;

async function onload() {
  const $clear = document.getElementById("clear");
  const $export = document.getElementById("export");
  const $output = document.getElementById("output");
  const $image = document.getElementById("image");

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

  const model = await tf.loadLayersModel("/model/model.json");
  $export.addEventListener(
    "click",
    async () => {
      // Interpret the handwriting.
      // Returns a tensor3d. Takes only 1 channel grayscale.
      let x = tf.browser
        .fromPixels(document.getElementById("canvas"), 1)
        .asType("float32");
      x = tf.image.resizeBilinear(x, [IMAGE_HEIGHT, IMAGE_WIDTH]);

      // The images have black background and white text.
      x = tf.sub(255, x);
      x = x.div(255); // Normalize.
      // For debugging.
      await tf.browser.toPixels(x, $image);

      // The tensor created by tf.browser.fromPixels does not include a batch dimension.
      // Becomes tensor4d
      x = x.expandDims();

      const output = model.predict(x);
      const axis = 1;
      const predicted = Array.from(output.argMax(axis).dataSync());
      $output.innerHTML = `Predicted: ${predicted[0]}`;
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
