package com.github.reinvent.the.wheel.cnn;


import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Created by shaoaq on 16-10-8.
 */
public class Demo {
    private static Logger logger = Logger.getLogger(Demo.class.getSimpleName());

    public static void main(String[] args) throws IOException, ParserConfigurationException, ClassNotFoundException {
        DataSet dataSet = new DataSet("dataSet/data.ds", 0.3);
        System.out.println(dataSet.getTrainSize());
        Cnn cnn = new CnnBuilder(50)
                .setInputLayer(new Size(28, 28))
                .addConvolutionalLayer(6, new Size(5, 5))
                .addSimpleLayer(new Size(2, 2))
                .addConvolutionalLayer(12, new Size(5, 5))
                .addSimpleLayer(new Size(2, 2))
                .setOutputLayer(10)
                .build();
        long now = System.currentTimeMillis();
        cnn.train(dataSet, 3);
        System.out.println("cost:" + (System.currentTimeMillis() - now));
        cnn.saveModel("demo.model");


        Cnn cnn1 = Cnn.readModel("demo.model");
        final int[] testRight = {0};
        final int[] testCount = {0};
        dataSet.testRecordForEach(record -> {
            if (cnn1.test(record)) {
                testRight[0]++;
            }
            testCount[0]++;
        });
        double testP = 1.0 * testRight[0] / testCount[0];
        logger.info("test precision " + testRight[0] + "/" + testCount[0] + "=" + testP);
    }
}
