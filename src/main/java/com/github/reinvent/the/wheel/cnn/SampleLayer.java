package com.github.reinvent.the.wheel.cnn;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Created by shaoaq on 16-10-9.
 */
public class SampleLayer implements Layer {
    private final Size scaleSize;
    private final int outCount;
    private final Layer preLayer;
    private final Size size;
    private final double[][][][] outs;
    private final double[][][][] errors;
    private Layer nextLayer;

    public SampleLayer(int batchSize, Size scaleSize, Layer preLayer) {
        this.scaleSize = scaleSize;
        this.outCount = preLayer.getOutCount();//采样层输出和上一层输出个数一致
        this.preLayer = preLayer;
        this.size = preLayer.getSize().divide(scaleSize);
        errors = new double[batchSize][outCount][size.getWidth()][size.getWidth()];
        outs = new double[batchSize][outCount][size.getWidth()][size.getHeight()];
    }

    public static double[][] scaleMatrix(final double[][] matrix,
                                         final Size scale) {
        int m = matrix.length;
        int n = matrix[0].length;
        final int sm = m / scale.getWidth();
        final int sn = n / scale.getHeight();
        final double[][] outMatrix = new double[sm][sn];
        if (sm * scale.getWidth() != m || sn * scale.getHeight() != n)
            throw new RuntimeException("scale不能整除matrix");
        final int size = scale.getWidth() * scale.getHeight();
        for (int i = 0; i < sm; i++) {
            for (int j = 0; j < sn; j++) {
                double sum = 0.0;
                for (int si = i * scale.getWidth(); si < (i + 1) * scale.getWidth(); si++) {
                    for (int sj = j * scale.getHeight(); sj < (j + 1) * scale.getHeight(); sj++) {
                        sum += matrix[si][sj];
                    }
                }
                outMatrix[i][j] = sum / size;
            }
        }
        return outMatrix;
    }

    @Override
    public void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function) {
        for (int i = 0; i < outCount; i++) {
            function.apply(i).accept(outs[recordIndex][i]);
        }
    }

    private void setOutput(int recordIndex, DataSet.Record record, Layer lastLayer) {
        //TODO: 可以并行化
        lastLayer.forEachOutput(recordIndex, i -> lastRectangle -> {
            outs[recordIndex][i] = scaleMatrix(lastRectangle, scaleSize);
        });
    }

    public Size getScaleSize() {
        return scaleSize;
    }

    @Override
    public Size getSize() {
        return size;
    }

    @Override
    public int getOutCount() {
        return outCount;
    }

    private boolean setErrors(int recordIndex, DataSet.Record record) {
        if (nextLayer instanceof LayerWithKernel) {
            forEachOutput(recordIndex, i -> out -> {
                final double[][][] sum = {null};// 对每一个卷积进行求和
                nextLayer.forEachOutput(recordIndex, j -> nextOut -> {
                    double[][] nextError = nextLayer.getError(recordIndex, j);
                    double[][] nextKernel = ((LayerWithKernel) nextLayer).getKernel(i, j);
                    if (sum[0] == null)
                        sum[0] = MathUtil.fullConvolutional(nextError, MathUtil.transRotate180(nextKernel));
                    else
                        sum[0] = MathUtil.trans(MathUtil.fullConvolutional(nextError, MathUtil.transRotate180(nextKernel)),
                                sum[0],
                                v1 -> v2 -> v1 + v2);
                });
                errors[recordIndex][i] = sum[0];
            });
        }
        return true;
    }

    @Override
    public void setNextLayer(Layer layer) {
        this.nextLayer = layer;
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
    public boolean backPropagation(DataSet.Record record) {
        return setErrors(record.getIndex(), record);
    }
}
