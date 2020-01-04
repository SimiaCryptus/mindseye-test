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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.util.Random;

public @com.simiacryptus.ref.lang.RefAware
class SupplementedProblemData implements ImageProblemData {

  private final int expansion = 10;
  private final ImageProblemData inner;
  private final Random random = new Random();

  public SupplementedProblemData(final ImageProblemData inner) {
    this.inner = inner;
  }

  public static void printSample(@Nonnull final NotebookOutput log, final Tensor[][] expanded, final int size) {
    @Nonnull final com.simiacryptus.ref.wrappers.RefArrayList<Tensor[]> list = new com.simiacryptus.ref.wrappers.RefArrayList<>(
        com.simiacryptus.ref.wrappers.RefArrays.asList(expanded));
    com.simiacryptus.ref.wrappers.RefCollections.shuffle(list);
    log.p("Expanded Training Data Sample: " + list.stream().limit(size).map(x -> {
      return log.png(x[0].toGrayImage(), "");
    }).reduce((a, b) -> a + b).get());
  }

  @Nullable
  protected static Tensor addNoise(@Nonnull final Tensor tensor) {
    return tensor.mapParallel((v) -> Math.random() < 0.9 ? v : v + Math.random() * 100);
  }

  protected static Tensor translate(final int dx, final int dy, @Nonnull final Tensor tensor) {
    final int sx = tensor.getDimensions()[0];
    final int sy = tensor.getDimensions()[1];
    return new Tensor(tensor.coordStream(true).mapToDouble(c -> {
      final int x = c.getCoords()[0] + dx;
      final int y = c.getCoords()[1] + dy;
      if (x < 0 || x >= sx) {
        return 0.0;
      } else if (y < 0 || y >= sy) {
        return 0.0;
      } else {
        return tensor.get(x, y);
      }
    }).toArray(), tensor.getDimensions());
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefStream<LabeledObject<Tensor>> trainingData() throws IOException {
    return inner.trainingData().flatMap(labeledObject -> {
      return com.simiacryptus.ref.wrappers.RefIntStream.range(0, expansion).mapToObj(i -> {
        final int dx = random.nextInt(10) - 5;
        final int dy = random.nextInt(10) - 5;
        return SupplementedProblemData.addNoise(SupplementedProblemData.translate(dx, dy, labeledObject.data));
      }).map(t -> new LabeledObject<>(t, labeledObject.label));
    });
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefStream<LabeledObject<Tensor>> validationData() throws IOException {
    return inner.validationData();
  }
}
