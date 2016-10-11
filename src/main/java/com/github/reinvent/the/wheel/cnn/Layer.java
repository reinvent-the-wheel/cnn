package com.github.reinvent.the.wheel.cnn;

import java.io.Serializable;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Created by shaoaq on 16-10-8.
 */
public interface Layer extends Serializable {

    Size getSize();

    int getOutCount();

    void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function);

    void setNextLayer(Layer layer);

    double[][] getError(int recordIndex, Integer outIndex);

    double[][] getOut(int recordIndex, int outIndex);

    void forward(DataSet.Record record);

    boolean backPropagation(DataSet.Record record);

}
