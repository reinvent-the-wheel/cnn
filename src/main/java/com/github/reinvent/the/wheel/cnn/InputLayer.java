package com.github.reinvent.the.wheel.cnn;


import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Created by shaoaq on 16-10-8.
 */
public class InputLayer implements Layer {
    private final Size size;
    /**
     * 每个输入为一个矩形
     */
    private final double[][][] outs;
    private Layer layer;

    public InputLayer(int batchSize, Size size) {
        this.size = size;
        this.outs = new double[batchSize][size.getWidth()][size.getHeight()];
    }

    private void setOutput(int index, double[] values) {
        for (int x = 0; x < size.getWidth(); x++) {
            for (int y = 0; y < size.getHeight(); y++) {
                outs[index][x][y] = values[size.getWidth() * x + y];
            }
        }
    }

    @Override
    public void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function) {
        function.apply(0).accept(outs[recordIndex]);
    }

    @Override
    public Size getSize() {
        return size;
    }

    @Override
    public int getOutCount() {
        return 1;
    }

    private boolean setErrors(int index, DataSet.Record record) {
        //no need to set error to input layer
        return true;
    }

    @Override
    public void setNextLayer(Layer layer) {

        this.layer = layer;
    }

    @Override
    public double[][] getError(int recordIndex, Integer outIndex) {
        throw new RuntimeException("no error for input layer");
    }

    @Override
    public double[][] getOut(int recordIndex, int outIndex) {
        return outs[recordIndex];
    }

    @Override
    public void forward(DataSet.Record record) {
        setOutput(record.getIndex(), record.getData());
    }

    @Override
    public boolean backPropagation(DataSet.Record record) {
        return setErrors(record.getIndex(), record);
    }
}
