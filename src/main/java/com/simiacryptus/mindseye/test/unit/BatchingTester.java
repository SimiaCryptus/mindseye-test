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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.SimpleListEval;
import com.simiacryptus.mindseye.test.SimpleResult;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.data.ScalarStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;

public @com.simiacryptus.ref.lang.RefAware class BatchingTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(BatchingTester.class);

  private final double tolerance;
  private int batchSize = 10;
  private final boolean validateDerivatives;

  public BatchingTester(final double tolerance, boolean validateDerivatives) {
    this.tolerance = tolerance;
    this.validateDerivatives = validateDerivatives;
  }

  public int getBatchSize() {
    return batchSize;
  }

  @Nonnull
  public BatchingTester setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  public double getRandom() {
    return 5 * (Math.random() - 0.5);
  }

  @Nonnull
  public ToleranceStatistics test(@Nullable final Layer reference, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference)
      return new ToleranceStatistics();

    final TensorList[] inputTensorLists = com.simiacryptus.ref.wrappers.RefArrays
        .stream(inputPrototype).map(t -> new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream
            .range(0, getBatchSize()).mapToObj(i -> t.map(v -> getRandom())).toArray(i -> new Tensor[i])))
        .toArray(i -> new TensorList[i]);
    @Nonnull
    final SimpleResult asABatch;
    final com.simiacryptus.ref.wrappers.RefList<SimpleEval> oneAtATime;
    {
      asABatch = SimpleListEval.run(reference, validateDerivatives, inputTensorLists);
      oneAtATime = com.simiacryptus.ref.wrappers.RefIntStream.range(0, getBatchSize()).mapToObj(batch -> {
        Tensor[] inputTensors = com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputTensorLists.length)
            .mapToObj(i -> inputTensorLists[i].get(batch)).toArray(i -> new Tensor[i]);
        return SimpleEval.run(reference, validateDerivatives, inputTensors);
      }).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    }

    TensorList batchOutput = asABatch.getOutput();
    @Nonnull
    IntFunction<ToleranceStatistics> toleranceStatisticsIntFunction = batch -> {
      @Nullable
      Tensor batchTensor = batchOutput.get(batch);
      return new ToleranceStatistics().accumulate(batchTensor.getData(), oneAtATime.get(batch).getOutput().getData());
    };
    int batchLength = batchOutput.length();
    @Nonnull
    final ToleranceStatistics outputAgreement = com.simiacryptus.ref.wrappers.RefIntStream
        .range(0, Math.min(getBatchSize(), batchLength)).mapToObj(toleranceStatisticsIntFunction)
        .reduce((a, b) -> a.combine(b)).get();
    if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
      logger.info("Batch Output: " + batchOutput.stream().map(x -> {
        return x.prettyPrint();
      }).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList()));
      logger.info("Singular Output: " + oneAtATime.stream().map(x -> x.getOutput().prettyPrint())
          .collect(com.simiacryptus.ref.wrappers.RefCollectors.toList()));
      throw new AssertionError("Output Corrupt: " + outputAgreement);
    }

    if (validateDerivatives) {
      ToleranceStatistics derivativeAgreement = com.simiacryptus.ref.wrappers.RefIntStream
          .range(0, Math.min(getBatchSize(), batchLength)).mapToObj(batch -> {
            IntFunction<ToleranceStatistics> statisticsFunction = input -> {
              @Nullable
              Tensor a = asABatch.getInputDerivative()[input].get(batch);
              Tensor b = oneAtATime.get(batch).getDerivative()[input];
              @Nonnull
              Tensor diff = a.minus(b);
              logger.info("Error: " + diff.prettyPrint());
              logger.info("Scalar Statistics: " + new ScalarStatistics().add(diff.getData()).getMetrics());
              double[][] points = com.simiacryptus.ref.wrappers.RefArrays.stream(diff.getData())
                  .mapToObj(x -> new double[] { x }).toArray(i -> new double[i][]);
              return new ToleranceStatistics().accumulate(a.getData(), b.getData());
            };
            return com.simiacryptus.ref.wrappers.RefIntStream.range(0, Math.min(inputPrototype.length, batchLength))
                .mapToObj(statisticsFunction).reduce((a, b) -> a.combine(b)).orElse(null);
          }).filter(x -> x != null).reduce((a, b) -> a.combine(b)).orElse(null);

      if (null != derivativeAgreement && !(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
        throw new AssertionError("Derivatives Corrupt: " + derivativeAgreement);
      }
      return null != derivativeAgreement ? derivativeAgreement.combine(outputAgreement) : outputAgreement;
    } else {
      return outputAgreement;
    }
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, final Layer reference,
      @Nonnull final Tensor... inputPrototype) {
    log.h1("Batch Execution");
    log.p(
        "Most layers, including this one, should behave the same no matter how the items are split between batches. We verify this:");
    return log.eval(() -> {
      return test(reference, inputPrototype);
    });
  }

  @Nonnull
  @Override
  public String toString() {
    return "BatchingTester{" + "tolerance=" + tolerance + ", batchSize=" + batchSize + '}';
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BatchingTester addRef() {
    return (BatchingTester) super.addRef();
  }

  public static @SuppressWarnings("unused") BatchingTester[] addRefs(BatchingTester[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BatchingTester::addRef)
        .toArray((x) -> new BatchingTester[x]);
  }

  public static @SuppressWarnings("unused") BatchingTester[][] addRefs(BatchingTester[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BatchingTester::addRefs)
        .toArray((x) -> new BatchingTester[x][]);
  }
}
