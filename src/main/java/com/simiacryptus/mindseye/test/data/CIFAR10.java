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
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.BinaryChunkIterator;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.test.LabeledObject;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.input.BoundedInputStream;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.zip.GZIPInputStream;

/**
 * The type Cifar 10.
 */
public class CIFAR10 {

  @Nullable
  private static final DataLoader<LabeledObject<Tensor>> training = new DataLoader<LabeledObject<Tensor>>() {
    {
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
    }

    @Override
    protected void read(@Nonnull final RefList<LabeledObject<Tensor>> queue) {
      try {
        @Nullable
        InputStream stream = null;
        try {
          stream = Util.cacheStream(TestUtil.S3_ROOT.resolve("cifar-10-binary.tar.gz"));
        } catch (@Nonnull NoSuchAlgorithmException | KeyManagementException e) {
          throw Util.throwException(e);
        }
        final int recordSize = 3073;
        @Nonnull final GZIPInputStream inflatedInput = new GZIPInputStream(stream);
        @Nullable final TarArchiveInputStream tar = new TarArchiveInputStream(inflatedInput);
        while (0 < inflatedInput.available()) {
          if (Thread.interrupted()) {
            break;
          }
          final TarArchiveEntry nextTarEntry = tar.getNextTarEntry();
          if (null == nextTarEntry) {
            break;
          }
          @Nonnull final BinaryChunkIterator iterator = new BinaryChunkIterator(
              new DataInputStream(new BoundedInputStream(tar, nextTarEntry.getSize())), recordSize);
          for (final byte[] chunk : (Iterable<byte[]>) () -> iterator) {
            LabeledObject<BufferedImage> bufferedImageLabeledObject = CIFAR10.toImage(chunk);
            queue.add(bufferedImageLabeledObject.map(Tensor::fromRGB));
            bufferedImageLabeledObject.freeRef();
          }
          iterator.freeRef();
        }
        System.err.println("Done loading");
      } catch (@Nonnull final IOException e) {
        e.printStackTrace();
        throw Util.throwException(e);
      }
      queue.freeRef();
    }
  };

  /**
   * Halt.
   */
  public static void halt() {
    assert CIFAR10.training != null;
    CIFAR10.training.stop();
  }

  /**
   * Training data stream ref stream.
   *
   * @return the ref stream
   */
  @Nonnull
  public static RefStream<LabeledObject<Tensor>> trainingDataStream() {
    assert CIFAR10.training != null;
    return CIFAR10.training.stream();
  }

  @Nonnull
  private static LabeledObject<BufferedImage> toImage(final byte[] b) {
    @Nonnull final BufferedImage img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < img.getWidth(); x++) {
      for (int y = 0; y < img.getHeight(); y++) {
        final int red = 0xFF & b[1 + 1024 * 0 + x + y * 32];
        final int blue = 0xFF & b[1 + 1024 * 1 + x + y * 32];
        final int green = 0xFF & b[1 + 1024 * 2 + x + y * 32];
        final int c = (red << 16) + (blue << 8) + green;
        img.setRGB(x, y, c);
      }
    }
    return new LabeledObject<>(img, RefArrays.toString(new byte[]{b[0]}));
  }

}
