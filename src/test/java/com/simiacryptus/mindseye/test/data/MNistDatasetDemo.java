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

import com.simiacryptus.lang.SupplierWeakCache;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public @RefAware
class MNistDatasetDemo extends ImageCategoryDatasetDemo {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return MNIST.class;
  }

  public static @SuppressWarnings("unused")
  MNistDatasetDemo[] addRefs(MNistDatasetDemo[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MNistDatasetDemo::addRef)
        .toArray((x) -> new MNistDatasetDemo[x]);
  }

  public static @SuppressWarnings("unused")
  MNistDatasetDemo[][] addRefs(MNistDatasetDemo[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MNistDatasetDemo::addRefs)
        .toArray((x) -> new MNistDatasetDemo[x][]);
  }

  @Override
  public RefStream<LabeledObject<SupplierWeakCache<BufferedImage>>> getTrainingStream(
      @Nonnull NotebookOutput log) {
    return log.eval(() -> {
      return MNIST.trainingDataStream().map(x -> x.map(y -> new SupplierWeakCache<>(() -> y.toImage())));
    });
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MNistDatasetDemo addRef() {
    return (MNistDatasetDemo) super.addRef();
  }
}
