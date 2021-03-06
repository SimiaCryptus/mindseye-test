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

import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.util.function.WeakCachedSupplier;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;

/**
 * The type M nist dataset demo.
 */
public class MNistDatasetDemo extends ImageCategoryDatasetDemo {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return MNIST.class;
  }

  @Override
  public RefStream<LabeledObject<WeakCachedSupplier<BufferedImage>>> getTrainingStream(@Nonnull NotebookOutput log) {
    return log.eval(() -> {
      return MNIST.trainingDataStream().map(x -> {
        LabeledObject<WeakCachedSupplier<BufferedImage>> map = x.map(y -> {
          WeakCachedSupplier<BufferedImage> temp_14_0001 = new WeakCachedSupplier<>(
              RefUtil.wrapInterface(y::toImage, y == null ? null : y.addRef()));
          if (null != y)
            y.freeRef();
          return temp_14_0001;
        });
        x.freeRef();
        return map;
      });
    });
  }

}
