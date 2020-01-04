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
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nullable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefStream;

public @com.simiacryptus.ref.lang.RefAware class CaltechProblemData implements ImageProblemData {

  private final int imageSize;
  @Nullable
  private com.simiacryptus.ref.wrappers.RefList<CharSequence> labels = null;

  public CaltechProblemData() {
    this(256);
  }

  public CaltechProblemData(int imageSize) {
    this.imageSize = imageSize;
  }

  public int getImageSize() {
    return imageSize;
  }

  @Nullable
  public com.simiacryptus.ref.wrappers.RefList<CharSequence> getLabels() {
    if (null == labels) {
      synchronized (this) {
        if (null == labels) {
          labels = trainingData().map(x -> x.label).distinct().sorted()
              .collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
        }
      }
    }
    return labels;
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefStream<LabeledObject<Tensor>> trainingData() {
    return Caltech101.trainingDataStream().parallel()
        .map(x -> x.map(y -> Tensor.fromRGB(ImageUtil.resize(y.get(), getImageSize()))));
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefStream<LabeledObject<Tensor>> validationData() {
    return trainingData();
  }

}
