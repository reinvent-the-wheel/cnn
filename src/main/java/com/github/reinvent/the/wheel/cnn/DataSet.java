package com.github.reinvent.the.wheel.cnn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Created by shaoaq on 16-10-8.
 */
public class DataSet {
    private final long size;
    private final long trainSize;
    private final long testSize;
    private final String fileName;
    private final Set<Long> testSelected;

    public DataSet(String fileName, double testRatio) throws IOException {
        this.fileName = fileName;
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            size = reader.lines().count();
        }
        this.testSize = (long) (size * testRatio);
        this.trainSize = size - testSize;
        testSelected = ThreadLocalRandom.current().longs(0, this.size).distinct().limit(testSize).boxed()
                .collect(Collectors.toSet());
    }

    public long getTrainSize() {
        return trainSize;
    }

    /**
     * 从DataSet训练数据里随机获取指定个数的数据执行
     *
     * @param size
     * @param consumer
     */
    void randomTrainRecordForEach(int size, Consumer<Record> consumer) throws IOException {
        Set<Long> selected = ThreadLocalRandom.current().longs(0, this.trainSize).distinct().limit(size).boxed()
                .collect(Collectors.toSet());
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            long lineNumber = 0L;
            long trainIndex = -1L;
            int recordIndex = 0;
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                if (!testSelected.contains(lineNumber)) {
                    trainIndex++;
                    if (selected.contains(trainIndex)) {
                        double[] doubles = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
                        Record record = new Record(new Double(doubles[0]).intValue(), recordIndex++, Arrays.copyOfRange(doubles, 1, doubles.length));
                        consumer.accept(record);
                    }
                }
                lineNumber++;
            }
        }
    }

    /**
     * 遍历DataSet的测试数据执行
     *
     * @param consumer
     */
    void testRecordForEach(Consumer<Record> consumer) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            long lineNumber = 0L;
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                if (testSelected.contains(lineNumber)) {
                    double[] doubles = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
                    Record record = new Record(new Double(doubles[0]).intValue(), 0, Arrays.copyOfRange(doubles, 1, doubles.length));
                    consumer.accept(record);
                }
                lineNumber++;
            }
        }
    }

    public static class Record {
        private final int label;
        private final int index;
        private final double[] data;

        public Record(int label, int index, double[] data) {
            this.label = label;
            this.index = index;
            this.data = data;
        }

        public int getIndex() {
            return index;
        }

        public double[] getData() {
            return data;
        }

        public int getLabel() {
            return label;
        }
    }
}
