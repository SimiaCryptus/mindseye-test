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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.BinaryChunkIterator;
import com.simiacryptus.util.io.DataLoader;
import com.simiacryptus.util.test.LabeledObject;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.*;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.zip.GZIPInputStream;

public @RefAware
class MNIST {

  public static final DataLoader<LabeledObject<Tensor>> training = new DataLoader<LabeledObject<Tensor>>() {
    {
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Override
    protected void read(@Nonnull final RefList<LabeledObject<Tensor>> queue) {
      try {
        final RefStream<Tensor> imgStream = MNIST.binaryStream("train-images-idx3-ubyte.gz", 16, 28 * 28).map(b -> {
          return MNIST.fillImage(b, new Tensor(28, 28, 1));
        });
        @Nonnull final RefStream<byte[]> labelStream = MNIST.binaryStream("train-labels-idx1-ubyte.gz", 8, 1);

        @Nonnull final RefStream<LabeledObject<Tensor>> merged = MNIST
            .toStream(new LabeledObjectIterator(imgStream, labelStream), 100);
        merged.forEach(RefUtil.wrapInterface(
            (Consumer<? super LabeledObject<Tensor>>) x -> queue
                .add(x),
            queue == null ? null : queue.addRef()));
      } catch (@Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
      queue.freeRef();
    }
  };
  public static final DataLoader<LabeledObject<Tensor>> validation = new DataLoader<LabeledObject<Tensor>>() {
    {
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    @Override
    protected void read(@Nonnull final RefList<LabeledObject<Tensor>> queue) {
      try {
        final RefStream<Tensor> imgStream = MNIST.binaryStream("t10k-images-idx3-ubyte.gz", 16, 28 * 28).map(b -> {
          return MNIST.fillImage(b, new Tensor(28, 28, 1));
        });
        @Nonnull final RefStream<byte[]> labelStream = MNIST.binaryStream("t10k-labels-idx1-ubyte.gz", 8, 1);

        @Nonnull final RefStream<LabeledObject<Tensor>> merged = MNIST
            .toStream(new LabeledObjectIterator(imgStream, labelStream), 100);
        merged.forEach(RefUtil.wrapInterface(
            (Consumer<? super LabeledObject<Tensor>>) x -> queue
                .add(x),
            queue == null ? null : queue.addRef()));
      } catch (@Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
      queue.freeRef();
    }
  };

  public static RefStream<LabeledObject<Tensor>> trainingDataStream() {
    return MNIST.training.stream();
  }

  public static RefStream<LabeledObject<Tensor>> validationDataStream() {
    return MNIST.validation.stream();
  }

  private static RefStream<byte[]> binaryStream(@Nonnull final String name, final int skip, final int recordSize)
      throws IOException {
    @Nullable
    InputStream stream = null;
    try {
      stream = Util.cacheStream(TestUtil.S3_ROOT.resolve(name));
    } catch (@Nonnull NoSuchAlgorithmException | KeyManagementException e) {
      throw new RuntimeException(e);
    }
    final byte[] fileData = IOUtils
        .toByteArray(new BufferedInputStream(new GZIPInputStream(new BufferedInputStream(stream))));
    @Nonnull final DataInputStream in = new DataInputStream(new ByteArrayInputStream(fileData));
    in.skip(skip);
    return MNIST.toIterator(new BinaryChunkIterator(in, recordSize));
  }

  @Nonnull
  private static Tensor fillImage(final byte[] b, @Nonnull final Tensor tensor) {
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        tensor.set(new int[]{x, y}, b[x + y * 28] & 0xFF);
      }
    }
    return tensor;
  }

  private static <T> RefStream<T> toIterator(@Nonnull final RefIteratorBase<T> iterator) {
    RefStream<T> temp_06_0003 = RefStreamSupport
        .stream(RefSpliterators.spliterator(iterator == null ? null : iterator, 1, Spliterator.ORDERED), false);
    return temp_06_0003;
  }

  private static <T> RefStream<T> toStream(@Nonnull final RefIteratorBase<T> iterator, final int size) {
    RefStream<T> temp_06_0004 = MNIST.toStream(iterator == null ? null : iterator, size,
        false);
    return temp_06_0004;
  }

  private static <T> RefStream<T> toStream(@Nonnull final RefIteratorBase<T> iterator, final int size,
                                           final boolean parallel) {
    RefStream<T> temp_06_0005 = RefStreamSupport
        .stream(RefSpliterators.spliterator(iterator == null ? null : iterator, size, Spliterator.ORDERED), parallel);
    return temp_06_0005;
  }

  private static class LabeledObjectIterator extends RefIteratorBase<LabeledObject<Tensor>> {
    @Nonnull
    private final RefIterator<Tensor> imgItr;
    @Nonnull
    private final RefIterator<byte[]> labelItr;

    public LabeledObjectIterator(RefStream<Tensor> imgStream, RefStream<byte[]> labelStream) {
      {
        RefIterator<Tensor> temp_06_0001 = imgStream
            .iterator();
        imgItr = temp_06_0001 == null ? null : temp_06_0001.addRef();
        if (null != temp_06_0001)
          temp_06_0001.freeRef();
      }
      {
        RefIterator<byte[]> temp_06_0002 = labelStream.iterator();
        labelItr = temp_06_0002 == null ? null : temp_06_0002.addRef();
        if (null != temp_06_0002)
          temp_06_0002.freeRef();
      }
    }

    public static @SuppressWarnings("unused")
    LabeledObjectIterator[] addRefs(LabeledObjectIterator[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(LabeledObjectIterator::addRef)
          .toArray((x) -> new LabeledObjectIterator[x]);
    }

    @Override
    public boolean hasNext() {
      return imgItr.hasNext() && labelItr.hasNext();
    }

    @Nonnull
    @Override
    public LabeledObject<Tensor> next() {
      return new LabeledObject<>(imgItr.next(), RefArrays.toString(labelItr.next()));
    }

    public @SuppressWarnings("unused")
    void _free() {
      labelItr.freeRef();
      imgItr.freeRef();
    }

    public @Override
    @SuppressWarnings("unused")
    LabeledObjectIterator addRef() {
      return (LabeledObjectIterator) super.addRef();
    }
  }

}
