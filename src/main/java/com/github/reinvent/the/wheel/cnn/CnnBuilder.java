package com.github.reinvent.the.wheel.cnn;


import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by shaoaq on 16-10-8.
 */
public class CnnBuilder {
    private final int batchSize;
    private InputLayerBuilder inputLayerBuilder;
    private OutputLayerBuilder outputLayerBuilder;
    private List<LayerBuilder> layerBuilders = new ArrayList<>();

    public CnnBuilder(int batchSize) {
        this.batchSize = batchSize;
    }

    public Cnn build() {
        InputLayer inputLayer = new InputLayer(batchSize, inputLayerBuilder.size);
        LinkedList<Layer> layers = new LinkedList<>();
        layers.add(inputLayer);
        Layer preLayer = inputLayer;
        for (LayerBuilder layerBuilder : layerBuilders) {
            Layer layer = layerBuilder.build(batchSize, preLayer);
            layers.add(layer);
            preLayer.setNextLayer(layer);
            preLayer = layer;
        }
        OutputLayer outputLayer = outputLayerBuilder.build(batchSize, preLayer);
        layers.add(outputLayer);
        preLayer.setNextLayer(outputLayer);
        return new Cnn(batchSize, layers);
    }

    public CnnBuilder setInputLayer(Size size) {
        this.inputLayerBuilder = new InputLayerBuilder(size);
        return this;
    }

    public CnnBuilder addConvolutionalLayer(int outRectangleCount, Size kernelSize) {
        layerBuilders.add(new ConvolutionalLayerBuilder(outRectangleCount, kernelSize));
        return this;
    }

    public CnnBuilder addSimpleLayer(Size size) {
        layerBuilders.add(new SimpleLayerBuilder(size));
        return this;
    }

    public CnnBuilder setOutputLayer(int classNum) {
        outputLayerBuilder = new OutputLayerBuilder(classNum);
        return this;
    }

    private interface LayerBuilder {
        Layer build(int batchSize, Layer lastLayer);
    }

    private static class InputLayerBuilder implements LayerBuilder {
        private final Size size;

        public InputLayerBuilder(Size size) {
            this.size = size;
        }

        @Override
        public Layer build(int batchSize, Layer lastLayer) {
            return new InputLayer(batchSize, size);
        }
    }

    private static class ConvolutionalLayerBuilder implements LayerBuilder {
        private final int outRectangleCount;
        private final Size kernelSize;

        public ConvolutionalLayerBuilder(int outRectangleCount, Size kernelSize) {
            this.outRectangleCount = outRectangleCount;
            this.kernelSize = kernelSize;
        }

        @Override
        public Layer build(int batchSize, Layer lastLayer) {
            return new ConvolutionalLayer(batchSize, outRectangleCount, kernelSize, lastLayer);
        }
    }

    private static class SimpleLayerBuilder implements LayerBuilder {
        private final Size scaleSize;

        public SimpleLayerBuilder(Size scaleSize) {
            this.scaleSize = scaleSize;
        }

        @Override
        public Layer build(int batchSize, Layer lastLayer) {
            return new SampleLayer(batchSize, scaleSize, lastLayer);
        }
    }

    private static class OutputLayerBuilder implements LayerBuilder {
        private final int classNum;

        public OutputLayerBuilder(int classNum) {
            this.classNum = classNum;
        }

        @Override
        public OutputLayer build(int batchSize, Layer lastLayer) {
            return new OutputLayer(batchSize, classNum, lastLayer);
        }
    }
}
