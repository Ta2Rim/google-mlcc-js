'use strict';
(function() {
    async function getData() {
        const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
        const carsData = await carsDataReq.json();
        const cleaned = carsData.map((car) => ({
            mpg : car.Miles_per_Gallon,
            horsepower : car.Horsepower
        }))
        .filter(car => (car.mpg != null && car.horsepower != null))
        return cleaned;
    }

    function createModel() {
        const model = tf.sequential(); // Create(instantiate) a sequential model

        //dense : a type of layer that multiplies its inputs by a matrix (called weights) and then adds a number(called the bias) to the result
        //inputShape is now [1] becuase we have 1 number as our input(the horsepower of a given car)

        model.add(tf.layers.dense({inputShape : [1], units : 1, useBias : true})); // Add a single hidden layer
        model.add(tf.layers.dense({units : 1, useBias : true})); //Use an output layer

        //We sets units to 1 because we want to output 1 number

        return model;
    }

    /**
     * Convert the input data to tensors that we can use for machine learning.
     * We will also do the important best practices of _shuffling_
     * the data and _normalizing_ the data MPG on the y-axis
     */
    function convertToTensor(data) {
        return tf.tidy(() => {
            // Step 1. Shuffle the data
            tf.util.shuffle(data);

            // Step 2. Conver data to Tensor
            const inputs = data.map(d=>d.horsepower);
            const labels = data.map(d=>d.mpg);

            // tensor will have a shape of [num_examples, num_features_per_example]
            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            // Step 3. normalize data to the range 0~1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();
        
            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

            return {
                inputs : normalizedInputs,
                labels : normalizedLabels,
                inputMax,
                inputMin,
                labelMax,
                labelMin
            }
        });
    }

    async function trainModel(model, inputs, labels) {
        model.compile({
            optimizer : tf.train.adam(),
            loss : tf.losses.meanSquaredError,
            metrics : ['mse']
        });
        // optimizer is the algorithm that is going to the updates to the model as it sees examples
        // loss is a function that will tell the model how well it is doing on learning each of batches that it is shown

        const batchSize = 28;
        const epochs = 50;

        // call to start the training loop
        return model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle : true,
            callbacks : tfvis.show.fitCallbacks(
                {name : 'Traning Performance'},
                ['loss'],
                { height : 200, callbacks : ['onEpochEnd']}
            )
        })
    }

    function testModel(model, inputData, normalizationData) {
        const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
        const [xs, preds] = tf.tidy(() => {
            //We generate 100 new ‘examples' to feed to the mode
            const xs = tf.linspace(0, 1, 100);

            // [num_examples, num_features_per_example]
            const preds = model.predict(xs.reshape([100, 1]));

            // unNormailze the Data to back to our original range(not 0~1)
            const unNormXs = xs
                .mul(inputMax.sub(inputMin))
                .add(inputMin);

            const unNormPreds = preds
                .mul(labelMax.sub(labelMin))
                .add(labelMin);

            //typedarray 
            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });

        const predictedPoints = Array.from(xs).map((val, i) => {
            return {x:val, y:preds[i]}
        })

        const originalPoints = inputData.map(d => ({
           x : d.horsepower, y: d.mpg 
        }));

        tfvis.render.scatterplot(
            {name : 'Model Predictions vs Original data'},
            {values : [originalPoints, predictedPoints], series : ['original', 'predicted']},
            {
                xLabel : 'Horsepower',
                yLabel : 'MPG',
                height : 300
            }
        )
    }

    async function run() {
        const data = await getData();
        const values = data.map(d => ({
            x : d.horsepower,
            y : d.mpg
        }));

        tfvis.render.scatterplot(
            {name : 'Horsepower v MPG'},
            {values},
            {
                xLabel : 'Horsepower',
                yLabel : 'MPG',
                height : 300
            }
        );

        const model = createModel();
        tfvis.show.modelSummary({name : 'model summary'}, model);
        
        const tensorData = convertToTensor(data);
        const {inputs, labels} = tensorData;

        await trainModel(model, inputs, labels);
        console.log('Done traning');
        
        // Make some predictions using the model and compare them to the original data
        testModel(model, data, tensorData);
    }


    document.addEventListener('DOMContentLoaded', run);
})();