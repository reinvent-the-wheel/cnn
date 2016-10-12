package com.github.reinvent.the.wheel.cnn;


import java.io.*;
import java.util.LinkedList;
import java.util.logging.Logger;

/**
 * Created by shaoaq on 16-10-8.
 */
public class Cnn implements Serializable {
    public static final double LAMBDA = 0;
    public static double ALPHA = 0.85;
    private static Logger logger = Logger.getLogger(Cnn.class.getSimpleName());
    private final int batchSize;
    private final LinkedList<Layer> layers;

    public Cnn(int batchSize, LinkedList<Layer> layers) {
        this.batchSize = batchSize;
        this.layers = layers;
        layers.forEach(layer -> {
            logger.info(layer.getClass().getSimpleName() + ":outs:" + layer.getOutCount());
        });
    }

    public static Cnn readModel(String fileName) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            return (Cnn) in.readObject();
        }
    }

    public void train(DataSet dataSet, int trainCount) throws IOException {
        int batchCount = new Double(Math.ceil((dataSet.getTrainSize() + 0.0) / batchSize)).intValue();
        for (int i = 0; i < trainCount; i++) {
            final int[] right = {0};
            final int[] count = {0};
            for (int j = 0; j < batchCount; j++) {
                logger.fine("train count:" + i + "/" + trainCount + ",batch count:" + j + "/" + batchCount);
                dataSet.randomTrainRecordForEach(batchSize, record -> {
                    boolean result = train(record);
                    if (result) {
                        right[0]++;
                    }
                    count[0]++;
                });
                updateParameters();
            }
            double p = 1.0 * right[0] / count[0];
            if (i % 10 == 1 && p > 0.96) {//动态调整准学习速率
                ALPHA = 0.001 + ALPHA * 0.9;
                logger.info("Set alpha = " + ALPHA);
            }
            logger.info("train precision " + right[0] + "/" + count[0] + "=" + p);

            final int[] testRight = {0};
            final int[] testCount = {0};
            dataSet.testRecordForEach(record -> {
                if (test(record)) {
                    testRight[0]++;
                }
                testCount[0]++;
            });
            double testP = 1.0 * testRight[0] / testCount[0];
            logger.info("test precision " + testRight[0] + "/" + testCount[0] + "=" + testP);
        }
    }

    public boolean test(DataSet.Record record) {
        return predict(record) == record.getLabel();
    }

    public Integer predict(DataSet.Record record) {
        forward(record);
        if (layers.size() > 0) {
            Layer lastLayer = layers.get(layers.size() - 1);
            if (lastLayer instanceof OutputLayer) {
                return ((OutputLayer) lastLayer).getLabel(record.getIndex());
            }
        }
        return null;
    }

    private void updateParameters() {
        layers.stream().filter(layer -> layer instanceof LayerWithKernel)
                .forEach(layer -> {
                    ((LayerWithKernel) layer).updateKernels();
                    ((LayerWithKernel) layer).updateBias();
                });
    }

    private boolean train(DataSet.Record record) {
        forward(record);
        return backPropagation(record);
    }

    private boolean backPropagation(DataSet.Record record) {
        //反向传播，取输出层的结果
        final Boolean[] result = {null};
        layers.descendingIterator().forEachRemaining(layer -> {
            boolean layerResult = layer.backPropagation(record);
            result[0] = result[0] == null ? layerResult : result[0];
        });
        return result[0];
    }

    private void forward(DataSet.Record record) {
        for (Layer layer : layers) {
            layer.forward(record);
        }
    }

    public void saveModel(String fileName) throws IOException {
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(
                new FileOutputStream(fileName))) {
            objectOutputStream.writeObject(this);
        }
    }
}
