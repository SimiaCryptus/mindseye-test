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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.data.ScalarStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.function.IntFunction;

public class BatchingTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(BatchingTester.class);

  private final double tolerance;
  private final boolean validateDerivatives;
  private int batchSize = 10;

  public BatchingTester(final double tolerance, boolean validateDerivatives) {
    this.tolerance = tolerance;
    this.validateDerivatives = validateDerivatives;
  }

  public int getBatchSize() {
    return batchSize;
  }

  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }

  public double getRandom() {
    return 5 * (Math.random() - 0.5);
  }

  @Nonnull
  public ToleranceStatistics test(@Nullable final Layer reference, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference) {
      RefUtil.freeRefs(inputPrototype);
      return new ToleranceStatistics();
    }

    final TensorList[] inputTensorLists = RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(t -> {
      TensorArray temp_15_0001 = new TensorArray(RefIntStream.range(0, getBatchSize()).mapToObj(RefUtil
          .wrapInterface((IntFunction<? extends Tensor>) i -> t.map(v -> getRandom()), t == null ? null : t.addRef()))
          .toArray(i -> new Tensor[i]));
      if (null != t)
        t.freeRef();
      return temp_15_0001;
    }).toArray(i -> new TensorList[i]);
    @Nonnull final SimpleResult asABatch = SimpleListEval.run(reference.addRef(), validateDerivatives,
        RefUtil.addRefs(inputTensorLists));
    final RefList<SimpleEval> oneAtATime = RefIntStream.range(0, getBatchSize())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends SimpleEval>) batch -> {
          Tensor[] inputTensors = RefIntStream.range(0, inputTensorLists.length)
              .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> inputTensorLists[i].get(batch),
                  RefUtil.addRefs(inputTensorLists)))
              .toArray(i -> new Tensor[i]);
          SimpleEval temp_15_0002 = SimpleEval.run(reference.addRef(), validateDerivatives,
              RefUtil.addRefs(inputTensors));
          RefUtil.freeRefs(inputTensors);
          return temp_15_0002;
        }, RefUtil.addRefs(inputTensorLists), reference.addRef()))
        .collect(RefCollectors.toList());

    reference.freeRef();
    RefUtil.freeRefs(inputTensorLists);
    TensorList batchOutput = asABatch.getOutput();
    @Nonnull
    IntFunction<ToleranceStatistics> toleranceStatisticsIntFunction = RefUtil.wrapInterface(batch -> {
      assert batchOutput != null;
      @Nullable
      Tensor batchTensor = batchOutput.get(batch);
      SimpleEval temp_15_0008 = oneAtATime.get(batch);
      Tensor temp_15_0009 = temp_15_0008.getOutput();
      assert temp_15_0009 != null;
      ToleranceStatistics temp_15_0003 = new ToleranceStatistics().accumulate(batchTensor.getData(),
          temp_15_0009.getData());
      temp_15_0009.freeRef();
      temp_15_0008.freeRef();
      batchTensor.freeRef();
      return temp_15_0003;
    }, oneAtATime == null ? null : oneAtATime.addRef(), batchOutput == null ? null : batchOutput.addRef());
    assert batchOutput != null;
    int batchLength = batchOutput.length();
    @Nonnull final ToleranceStatistics outputAgreement = RefUtil.get(RefIntStream.range(0, Math.min(getBatchSize(), batchLength))
        .mapToObj(toleranceStatisticsIntFunction).reduce((a, b) -> a.combine(b)));
    if (!(outputAgreement.absoluteTol.getMax() < tolerance)) {
      RefList<String> temp_15_0010 = batchOutput.stream().map(x -> {
        String temp_15_0004 = x.prettyPrint();
        x.freeRef();
        return temp_15_0004;
      }).collect(RefCollectors.toList());
      logger.info("Batch Output: " + temp_15_0010);
      if (null != temp_15_0010)
        temp_15_0010.freeRef();
      assert oneAtATime != null;
      RefList<String> temp_15_0012 = oneAtATime.stream().map(x -> {
        Tensor temp_15_0011 = x.getOutput();
        assert temp_15_0011 != null;
        String temp_15_0005 = temp_15_0011.prettyPrint();
        temp_15_0011.freeRef();
        x.freeRef();
        return temp_15_0005;
      }).collect(RefCollectors.toList());
      logger.info("Singular Output: " + temp_15_0012);
      if (null != temp_15_0012)
        temp_15_0012.freeRef();
      asABatch.freeRef();
      oneAtATime.freeRef();
      batchOutput.freeRef();
      RefUtil.freeRefs(inputPrototype);
      throw new AssertionError("Output Corrupt: " + outputAgreement);
    }

    batchOutput.freeRef();
    if (validateDerivatives) {
      ToleranceStatistics derivativeAgreement = RefIntStream.range(0, Math.min(getBatchSize(), batchLength))
          .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) batch -> {
                TensorList[] asABatchInputDerivative = asABatch.getInputDerivative();
                assert oneAtATime != null;
                SimpleEval temp_15_0013 = oneAtATime.get(batch);
                Tensor[] derivative = temp_15_0013.getDerivative();
                temp_15_0013.freeRef();
                IntFunction<ToleranceStatistics> statisticsFunction = RefUtil.wrapInterface(input -> {
                  assert asABatchInputDerivative != null;
                  @Nullable
                  Tensor a = asABatchInputDerivative[input].get(batch);
                  assert derivative != null;
                  Tensor b = derivative[input].addRef();
                  @Nonnull
                  Tensor diff = a.minus(b.addRef());
                  logger.info("Error: " + diff.prettyPrint());
                  Map<CharSequence, Object> temp_15_0014 = new ScalarStatistics().add(diff.getData()).getMetrics();
                  logger.info("Scalar Statistics: " + temp_15_0014);
                  double[][] points = RefArrays.stream(diff.getData()).mapToObj(x -> new double[]{x})
                      .toArray(i -> new double[i][]);
                  diff.freeRef();
                  ToleranceStatistics temp_15_0006 = new ToleranceStatistics().accumulate(a.getData(), b.getData());
                  b.freeRef();
                  a.freeRef();
                  return temp_15_0006;
                }, oneAtATime.addRef(), asABatch.addRef(), asABatchInputDerivative, derivative);
                return RefIntStream.range(0, Math.min(inputPrototype.length, batchLength)).mapToObj(statisticsFunction)
                    .reduce((a, b) -> a.combine(b)).orElse(null);
              }, RefUtil.addRefs(inputPrototype), oneAtATime == null ? null : oneAtATime.addRef(),
              asABatch.addRef()))
          .filter(x -> x != null).reduce((a, b) -> a.combine(b)).orElse(null);

      if (null != derivativeAgreement && !(derivativeAgreement.absoluteTol.getMax() < tolerance)) {
        asABatch.freeRef();
        if (null != oneAtATime)
          oneAtATime.freeRef();
        RefUtil.freeRefs(inputPrototype);
        throw new AssertionError("Derivatives Corrupt: " + derivativeAgreement);
      }
      asABatch.freeRef();
      if (null != oneAtATime)
        oneAtATime.freeRef();
      RefUtil.freeRefs(inputPrototype);
      return null != derivativeAgreement ? derivativeAgreement.combine(outputAgreement) : outputAgreement;
    } else {
      asABatch.freeRef();
      if (null != oneAtATime)
        oneAtATime.freeRef();
      RefUtil.freeRefs(inputPrototype);
      return outputAgreement;
    }
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nullable final Layer reference,
                                  @Nonnull final Tensor... inputPrototype) {
    log.h1("Batch Execution");
    log.p(
        "Most layers, including this one, should behave the same no matter how the items are split between batches. We verify this:");
    ToleranceStatistics temp_15_0007 = log.eval(RefUtil.wrapInterface(() -> {
      return test(reference == null ? null : reference.addRef(), RefUtil.addRefs(inputPrototype));
    }, RefUtil.addRefs(inputPrototype), reference == null ? null : reference.addRef()));
    RefUtil.freeRefs(inputPrototype);
    if (null != reference)
      reference.freeRef();
    return temp_15_0007;
  }

  @Nonnull
  @Override
  public String toString() {
    return "BatchingTester{" + "tolerance=" + tolerance + ", batchSize=" + batchSize + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BatchingTester addRef() {
    return (BatchingTester) super.addRef();
  }
}
