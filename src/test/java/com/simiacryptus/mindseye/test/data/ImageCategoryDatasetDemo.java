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

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.function.WeakCachedSupplier;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.NotebookTestBase;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.awt.image.BufferedImage;

/**
 * The type Image category dataset demo.
 */
public abstract class ImageCategoryDatasetDemo extends NotebookTestBase {
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Data;
  }

  /**
   * Gets shuffle comparator.
   *
   * @param <T> the type parameter
   * @return the shuffle comparator
   */
  @Nonnull
  public <T> RefComparator<T> getShuffleComparator() {
    final int seed = (int) ((RefSystem.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return RefComparator.comparingInt(a1 -> RefSystem.identityHashCode(a1) ^ seed);
  }

  /**
   * Run.
   */
  @Test
  public void run() {
    @Nonnull NotebookOutput log = getLog();
    log.h3("Loading Data");
    RefList<LabeledObject<WeakCachedSupplier<BufferedImage>>> testData = getTrainingStream(log)
        .sorted(getShuffleComparator()).collect(RefCollectors.toList());

    log.h3("Categories");
    log.run(RefUtil.wrapInterface(() -> {
      RefMap<String, Long> temp_22_0001 = testData.stream()
          .collect(RefCollectors.groupingBy(x -> {
            String label = x.label;
            x.freeRef();
            return label;
          }, RefCollectors.counting()));
      temp_22_0001.forEach((k, v) -> ImageCategoryDatasetDemo.logger.info(RefString.format("%s -> %d", k, v)));
      temp_22_0001.freeRef();
    }, testData == null ? null : testData.addRef()));

    log.h3("Sample Data");
    log.p(log.out(RefUtil.wrapInterface((UncheckedSupplier<String>) () -> {
      assert testData != null;
      return RefUtil.get(testData.stream().map(labeledObj -> {
        @Nullable
        BufferedImage img = labeledObj.data.get();
        String label = labeledObj.label;
        labeledObj.freeRef();
        assert img != null;
        img = ImageUtil.resize(img, 224, true);
        return log.png(img, label);
      }).limit(20).reduce((a, b) -> a + b));
    }, testData == null ? null : testData.addRef())));
    if (null != testData)
      testData.freeRef();
  }

  /**
   * Gets training stream.
   *
   * @param log the log
   * @return the training stream
   */
  public abstract RefStream<LabeledObject<WeakCachedSupplier<BufferedImage>>> getTrainingStream(NotebookOutput log);

}
