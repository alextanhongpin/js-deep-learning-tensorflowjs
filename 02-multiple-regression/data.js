export class BostonDataset {
  constructor() {
    this.paths = [
      "./data/test-data.csv",
      "./data/test-target.csv",
      "./data/train-data.csv",
      "./data/train-target.csv",
    ];
    this.data = [];
  }

  get numFeatures() {
    if (!this.#loaded)
      throw new Error("BostonDatasetError: dataset not loaded");
    const [[trainX]] = this.data;
    return trainX[0].length;
  }

  get #loaded() {
    return this.data.length;
  }

  async load() {
    if (this.#loaded) return;

    const [trainX, trainY, testX, testY] = await Promise.all(
      this.paths.map(async (path) => {
        const body = await fetch(path);
        const text = await body.text();
        const [header, ...rows] = text.trim().split("\n");
        return rows.map((row) => row.split(",").map(parseFloat));
      })
    );

    this.data = [
      [trainX, trainY],
      [testX, testY],
    ];

    return this.data;
  }
}
