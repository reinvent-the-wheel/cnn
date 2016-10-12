package com.github.reinvent.the.wheel.cnn;

import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Created by shaoaq on 16-10-8.
 */
public class ConvolutionalLayer implements LayerWithKernel {
    private final Size size;
    private final double[][][][] outs;
    private final double[][][][] kernels;
    private final double[][][][] errors;
    private final double[] bias;
    private final int outCount;
    private final Layer preLayer;
    private final int batchSize;
    private Layer nextLayer;


    public ConvolutionalLayer(int batchSize, int outCount, Size kernelSize, Layer preLayer) {
        this.batchSize = batchSize;
        this.outCount = outCount;
        this.preLayer = preLayer;
        size = preLayer.getSize().subtract(kernelSize, 1);
        kernels = new double[preLayer.getOutCount()][outCount][kernelSize.getWidth()][kernelSize.getHeight()];
        //Kernels初始化为随机数?
        Random random = new Random();
        for (int i = 0; i < kernels.length; i++) {
            for (int j = 0; j < kernels[0].length; j++) {
                for (int k = 0; k < kernels[0][0].length; k++) {
                    for (int l = 0; l < kernels[0][0][0].length; l++) {
                        kernels[i][j][k][l] = (random.nextDouble() - 0.5) / 10;
                    }
                }
            }
        }
        bias = new double[outCount];
        //TODO:bias初始化为随机数?
        errors = new double[batchSize][outCount][size.getWidth()][size.getWidth()];
        outs = new double[batchSize][outCount][size.getWidth()][size.getHeight()];
    }

    public int getOutCount() {
        return outCount;
    }

    private boolean setErrors(int recordIndex, DataSet.Record record) {
        //TODO: 可以并行化
        if (nextLayer instanceof SampleLayer) {
            Size scaleSize = ((SampleLayer) nextLayer).getScaleSize();
            forEachOutput(recordIndex, i -> out -> {
                double[][] nextError = nextLayer.getError(recordIndex, i);
                errors[recordIndex][i] = MathUtil.trans(
                        MathUtil.trans(out,
                                MathUtil.trans(out, v -> 1 - v), v1 -> v2 -> v1 * v2
                        ),
                        MathUtil.scale(nextError, scaleSize),
                        v1 -> v2 -> v1 * v2
                );
            });

        } else {
            throw new RuntimeException("can only support sample layer after convolutional layer");
        }
        return true;
    }

    @Override
    public double[][] getError(int recordIndex, Integer outIndex) {
        return errors[recordIndex][outIndex];
    }

    @Override
    public double[][] getOut(int recordIndex, int outIndex) {
        return outs[recordIndex][outIndex];
    }

    @Override
    public void forward(DataSet.Record record) {
        setOutput(record.getIndex(), record, preLayer);
    }

    @Override
    public void setNextLayer(Layer layer) {
        this.nextLayer = layer;
    }

    @Override
    public void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function) {
        for (int i = 0; i < outCount; i++) {
            function.apply(i).accept(outs[recordIndex][i]);
        }
    }

    //    @Override
    private void setOutput(int recordIndex, DataSet.Record record, Layer lastLayer) {
        //TODO: 可以并行化
        for (int j = 0; j < outs[recordIndex].length; j++) {
            final int finalJ = j;
            final double[][][] sum = {null};// 对每一个输入map的卷积进行求和
            lastLayer.forEachOutput(recordIndex, i -> lastRectangle -> {
                double[][] kernel = kernels[i][finalJ];
                if (sum[0] == null) {
                    sum[0] = MathUtil.validConvolutional(lastRectangle, kernel);
                } else {
                    sum[0] = MathUtil.trans(MathUtil.validConvolutional(lastRectangle, kernel), sum[0], v1 -> v2 -> v1 + v2);
//                    sum[0] = matrixPlus(validConvolutional(lastRectangle, kernel), sum[0]);
                }
            });
            final double bias = this.bias[j];
//            sum[0] = matrixOp(sum[0], value -> sigmoid(value + bias));
            sum[0] = MathUtil.trans(sum[0], value -> MathUtil.sigmoid(value + bias));
            outs[recordIndex][j] = sum[0];
        }
    }

    @Override
    public Size getSize() {
        return size;
    }

    @Override
    public double[][] getKernel(int preLayerOutIndex, int thisLayerOutIndex) {
        return kernels[preLayerOutIndex][thisLayerOutIndex];
    }

    @Override
    public void updateKernels() {
        for (int j = 0; j < outCount; j++) {
            for (int i = 0; i < preLayer.getOutCount(); i++) {
                // 对batch的每个记录delta求和
                double[][] deltaKernel = null;
                for (int r = 0; r < batchSize; r++) {
                    double[][] error = errors[r][j];
                    if (deltaKernel == null) {
                        deltaKernel = MathUtil.validConvolutional(preLayer.getOut(r, i), error);
                    } else {// 累积求和
                        deltaKernel = MathUtil.trans(
                                deltaKernel,
                                MathUtil.validConvolutional(preLayer.getOut(r, i), error),
                                v1 -> v2 -> v1 + v2);
                    }
                }
                deltaKernel = MathUtil.trans(deltaKernel, v -> v / batchSize);
                // 更新卷积核
                double[][] kernel = kernels[i][j];
                kernel = MathUtil.trans(MathUtil.trans(kernel, v -> v * (1 - Cnn.LAMBDA * Cnn.ALPHA))
                        , MathUtil.trans(deltaKernel, v -> v * Cnn.ALPHA),
                        v1 -> v2 -> v1 + v2);
                kernels[i][j] = kernel;
            }
        }
    }

    @Override
    public void updateBias() {
        for (int j = 0; j < outCount; j++) {
            double[][] error = OutputLayer.sum(errors, j);
            double deltaBias = OutputLayer.sum(error) / batchSize;
            double bias = this.bias[j] + Cnn.ALPHA * deltaBias;
            this.bias[j] = bias;
        }
    }

    @Override
    public boolean backPropagation(DataSet.Record record) {
        return setErrors(record.getIndex(), record);
    }
}
