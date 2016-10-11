package com.github.reinvent.the.wheel.cnn;

/**
 * Created by shaoaq on 16-10-8.
 */
public interface LayerWithKernel extends Layer {
    double[][] getKernel(int preLayerOutIndex, int thisLayerOutIndex);

    void updateKernels();

    void updateBias();
}
