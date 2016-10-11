package com.github.reinvent.the.wheel.cnn;

/**
 * Created by shaoaq on 16-10-8.
 */
public class Rectangle {
    private final double[][] values;

    public Rectangle(int width, int height) {
        values = new double[width][height];
    }

    public void set(int x, int y, double value) {
        values[x][y] = value;
    }
}
