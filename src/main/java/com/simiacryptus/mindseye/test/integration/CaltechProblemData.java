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
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.stream.Collectors;

public class CaltechProblemData implements ImageProblemData {

  private final int imageSize;
  @Nullable
  private List<CharSequence> labels = null;

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
  public List<CharSequence> getLabels() {
    if (null == labels) {
      synchronized (this) {
        if (null == labels) {
          labels = trainingData().map(x -> {
            String label = x.label;
            x.freeRef();
            return label;
          }).distinct().sorted().collect(Collectors.toList());
        }
      }
    }
    return labels;
  }

  @Nonnull
  @Override
  public RefStream<LabeledObject<Tensor>> trainingData() {
    return Caltech101.trainingDataStream().parallel()
        .map(x -> {
          LabeledObject<Tensor> map = x.map(y -> {
            BufferedImage image = y.get();
            Tensor tensor = Tensor.fromRGB(ImageUtil.resize(image, getImageSize()));
            y.freeRef();
            return tensor;
          });
          x.freeRef();
          return map;
        });
  }

  @Nonnull
  @Override
  public RefStream<LabeledObject<Tensor>> validationData() {
    return trainingData();
  }

}
