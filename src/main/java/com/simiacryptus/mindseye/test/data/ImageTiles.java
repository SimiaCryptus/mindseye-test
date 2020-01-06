/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test.data;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.io.DataLoader;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public @RefAware
class ImageTiles {

  @Nonnull
  public static Tensor read(@Nonnull final BufferedImage image, final int width, final int height, final int x,
                            final int y) {
    @Nonnull final Tensor tensor = new Tensor(width, height, 3);
    for (int xx = 0; xx < width; xx++) {
      for (int yy = 0; yy < height; yy++) {
        @Nonnull final Color rgb = new Color(image.getRGB(x + xx, y + yy));
        tensor.set(new int[]{xx, yy, 0}, rgb.getRed());
        tensor.set(new int[]{xx, yy, 1}, rgb.getGreen());
        tensor.set(new int[]{xx, yy, 2}, rgb.getBlue());
      }
    }
    return tensor;
  }

  public static Stream<File> readFiles(@Nonnull final File dir) {
    if (dir.isFile()) {
      return Arrays.asList(dir).stream();
    }
    return Arrays.stream(dir.listFiles()).flatMap(ImageTiles::readFiles);
  }

  public static Tensor[] tilesRgb(@Nonnull final BufferedImage image, final int width, final int height) {
    return ImageTiles.tilesRgb(image, width, height, false);
  }

  public static Tensor[] tilesRgb(@Nonnull final BufferedImage image, final int width, final int height,
                                  final boolean overlap) {
    return ImageTiles.tilesRgb(image, width, height, overlap ? 1 : width, overlap ? 1 : height);
  }

  public static Tensor[] tilesRgb(@Nonnull final BufferedImage image, final int width, final int height,
                                  final int xStep, final int yStep) {
    @Nonnull final RefList<Tensor> tensors = new RefArrayList<>();
    for (int y = 0; y < image.getHeight(); y += yStep) {
      for (int x = 0; x < image.getWidth(); x += xStep) {
        try {
          @Nonnull final Tensor tensor = ImageTiles.read(image, width, height, y, x);
          tensors.add(tensor == null ? null : tensor);
        } catch (@Nonnull final ArrayIndexOutOfBoundsException e) {
          // Ignore
        }
      }
    }
    Tensor[] temp_17_0001 = tensors.toArray(new Tensor[]{});
    tensors.freeRef();
    return temp_17_0001;
  }

  @Nonnull
  public static RefList<Tensor> toTiles(@Nullable final BufferedImage image, final int tileWidth, final int tileHeight,
                                        final int minSpacingWidth, final int minSpacingHeight, final int maxTileCols, final int maxTileRows) {
    @Nonnull final RefList<Tensor> queue = new RefArrayList<>();
    if (null != image) {
      final int xMax = image.getWidth() - tileWidth;
      final int yMax = image.getHeight() - tileHeight;
      final int cols = Math.min(maxTileCols, xMax / minSpacingWidth);
      final int rows = Math.min(maxTileRows, yMax / minSpacingHeight);
      if (cols < 1) {
        return queue;
      }
      if (rows < 1) {
        return queue;
      }
      final int xStep = xMax / cols;
      final int yStep = yMax / rows;
      for (int x = 0; x < xMax; x += xStep) {
        for (int y = 0; y < yMax; y += yStep) {
          queue.add(ImageTiles.read(image, tileWidth, tileHeight, x, y));
        }
      }
    }
    return queue;
  }

  @Nonnull
  public static RefList<Tensor> toTiles(@Nonnull final File file, final int tileWidth, final int tileHeight,
                                        final int minSpacingWidth, final int minSpacingHeight, final int maxTileCols, final int maxTileRows)
      throws IOException {
    return ImageTiles.toTiles(ImageIO.read(file), tileWidth, tileHeight, minSpacingWidth, minSpacingHeight, maxTileCols,
        maxTileRows);
  }

  public static @RefAware
  class ImageTensorLoader extends DataLoader<Tensor> {

    public final int maxTileCols;
    public final int maxTileRows;
    public final int minSpacingHeight;
    public final int minSpacingWidth;
    public final File parentDirectiory;
    public final int tileHeight;
    public final int tileWidth;

    public ImageTensorLoader(final File parentDirectiory, final int tileWidth, final int tileHeight,
                             final int minSpacingWidth, final int minSpacingHeight, final int maxTileRows, final int maxTileCols) {
      this.parentDirectiory = parentDirectiory;
      this.tileWidth = tileWidth;
      this.tileHeight = tileHeight;
      this.minSpacingWidth = minSpacingWidth;
      this.minSpacingHeight = minSpacingHeight;
      this.maxTileRows = maxTileRows;
      this.maxTileCols = maxTileCols;
    }

    public static @SuppressWarnings("unused")
    ImageTensorLoader[] addRefs(ImageTensorLoader[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(ImageTensorLoader::addRef)
          .toArray((x) -> new ImageTensorLoader[x]);
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    ImageTensorLoader addRef() {
      return (ImageTensorLoader) super.addRef();
    }

    @Override
    protected void read(@Nonnull final RefList<Tensor> queue) {
      @Nonnull final ArrayList<File> files = new ArrayList<>(
          ImageTiles.readFiles(parentDirectiory).collect(Collectors.toList()));
      Collections.shuffle(files);
      for (@Nonnull final File f : files) {
        if (Thread.interrupted()) {
          break;
        }
        try {
          queue.addAll(ImageTiles.toTiles(f, tileWidth, tileHeight, minSpacingWidth, minSpacingHeight, maxTileCols,
              maxTileRows));
        } catch (@Nonnull final Throwable e) {
          throw new RuntimeException(e);
        }
      }
      queue.freeRef();
    }
  }
}
