package com.github.reinvent.the.wheel.cnn;

import java.io.Serializable;

/**
 * Created by shaoaq on 16-10-8.
 */
public class Size implements Serializable {
    private final int width;
    private final int height;

    public Size(int width, int height) {
        this.width = width;
        this.height = height;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public Size subtract(Size size, int append) {
        int width = this.width - size.width + append;
        int height = this.height - size.height + append;
        return new Size(width, height);
    }

    public Size divide(Size scaleSize) {
        int width = this.width / scaleSize.width;
        int height = this.height / scaleSize.height;
        if (width * scaleSize.width != this.width || height * scaleSize.height != this.height)
            throw new RuntimeException("invalidate scale size");
        return new Size(width, height);
    }
}
