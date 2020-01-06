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
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.test.LabeledObject;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public abstract @RefAware
class ImageCategoryDatasetDemo extends NotebookReportBase {
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Data;
  }

  public <T> RefComparator<T> getShuffleComparator() {
    final int seed = (int) ((com.simiacryptus.ref.wrappers.RefSystem.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return RefComparator.comparingInt(a1 -> com.simiacryptus.ref.wrappers.RefSystem.identityHashCode(a1) ^ seed);
  }

  public static @SuppressWarnings("unused")
  ImageCategoryDatasetDemo[] addRefs(ImageCategoryDatasetDemo[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImageCategoryDatasetDemo::addRef)
        .toArray((x) -> new ImageCategoryDatasetDemo[x]);
  }

  public static @SuppressWarnings("unused")
  ImageCategoryDatasetDemo[][] addRefs(ImageCategoryDatasetDemo[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImageCategoryDatasetDemo::addRefs)
        .toArray((x) -> new ImageCategoryDatasetDemo[x][]);
  }

  @Test
  public void run() {
    run(this::run);
  }

  public void run(@Nonnull NotebookOutput log) {
    log.h3("Loading Data");
    RefList<LabeledObject<SupplierWeakCache<BufferedImage>>> testData = getTrainingStream(log)
        .sorted(getShuffleComparator()).collect(RefCollectors.toList());

    log.h3("Categories");
    log.run(RefUtil.wrapInterface(() -> {
      RefMap<String, Long> temp_22_0001 = testData.stream()
          .collect(RefCollectors.groupingBy(x -> x.label, RefCollectors.counting()));
      temp_22_0001.forEach((k, v) -> ImageCategoryDatasetDemo.logger.info(RefString.format("%s -> %d", k, v)));
      if (null != temp_22_0001)
        temp_22_0001.freeRef();
    }, testData == null ? null : testData.addRef()));

    log.h3("Sample Data");
    log.p(log.out(RefUtil
        .wrapInterface((UncheckedSupplier<String>) () -> {
          return testData.stream().map(labeledObj -> {
            @Nullable
            BufferedImage img = labeledObj.data.get();
            img = ImageUtil.resize(img, 224, true);
            return log.png(img, labeledObj.label);
          }).limit(20).reduce((a, b) -> a + b).get();
        }, testData == null ? null : testData.addRef())));
    if (null != testData)
      testData.freeRef();
  }

  public abstract RefStream<LabeledObject<SupplierWeakCache<BufferedImage>>> getTrainingStream(NotebookOutput log);

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImageCategoryDatasetDemo addRef() {
    return (ImageCategoryDatasetDemo) super.addRef();
  }
}
