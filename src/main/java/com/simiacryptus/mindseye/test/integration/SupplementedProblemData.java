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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.util.Random;
import java.util.function.IntFunction;
import java.util.function.ToDoubleFunction;

/**
 * The type Supplemented problem data.
 */
public class SupplementedProblemData implements ImageProblemData {

  private final int expansion = 10;
  private final ImageProblemData inner;
  private final Random random = new Random();

  /**
   * Instantiates a new Supplemented problem data.
   *
   * @param inner the inner
   */
  public SupplementedProblemData(final ImageProblemData inner) {
    this.inner = inner;
  }

  /**
   * Print sample.
   *
   * @param log      the log
   * @param expanded the expanded
   * @param size     the size
   */
  public static void printSample(@Nonnull final NotebookOutput log, @Nullable final Tensor[][] expanded, final int size) {
    @Nonnull final RefArrayList<Tensor[]> list = new RefArrayList<>(RefArrays.asList(expanded));
    RefCollections.shuffle(list.addRef());
    log.p("Expanded Training Data Sample: " + RefUtil.get(list.stream().limit(size).map(x -> {
      String temp_16_0001 = log.png(x[0].toGrayImage(), "");
      RefUtil.freeRef(x);
      return temp_16_0001;
    }).reduce((a, b) -> a + b)));
    list.freeRef();
  }

  /**
   * Add noise tensor.
   *
   * @param tensor the tensor
   * @return the tensor
   */
  @Nonnull
  protected static Tensor addNoise(@Nonnull final Tensor tensor) {
    Tensor temp_16_0003 = tensor.mapParallel(v -> Math.random() < 0.9 ? v : v + Math.random() * 100);
    tensor.freeRef();
    return temp_16_0003;
  }

  /**
   * Translate tensor.
   *
   * @param dx     the dx
   * @param dy     the dy
   * @param tensor the tensor
   * @return the tensor
   */
  @Nonnull
  protected static Tensor translate(final int dx, final int dy, @Nonnull final Tensor tensor) {
    final int sx = tensor.getDimensions()[0];
    final int sy = tensor.getDimensions()[1];
    return new Tensor(
        tensor.coordStream(true).mapToDouble(RefUtil.wrapInterface((ToDoubleFunction<? super Coordinate>) c -> {
          final int x = c.getCoords()[0] + dx;
          final int y = c.getCoords()[1] + dy;
          if (x < 0 || x >= sx) {
            return 0.0;
          } else if (y < 0 || y >= sy) {
            return 0.0;
          } else {
            return tensor.get(x, y);
          }
        }, tensor)).toArray(), tensor.getDimensions());
  }

  @Nonnull
  @Override
  public RefStream<LabeledObject<Tensor>> trainingData() throws IOException {
    return inner.trainingData().flatMap(labeledObject -> {
      return RefIntStream.range(0, expansion)
          .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) i -> {
            final int dx = random.nextInt(10) - 5;
            final int dy = random.nextInt(10) - 5;
            return SupplementedProblemData.addNoise(SupplementedProblemData.translate(dx, dy, labeledObject.data.addRef()));
          }, labeledObject.addRef()))
          .map(RefUtil.wrapInterface(t -> {
            LabeledObject<Tensor> temp_16_0002 = new LabeledObject<>(t.addRef(), labeledObject.label);
            t.freeRef();
            return temp_16_0002;
          }, labeledObject));
    });
  }

  @Override
  public RefStream<LabeledObject<Tensor>> validationData() throws IOException {
    return inner.validationData();
  }
}
