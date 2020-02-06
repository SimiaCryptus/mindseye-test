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

import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.data.DoubleStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.function.IntFunction;

public class PerformanceTester extends ComponentTestBase<ToleranceStatistics> {
  static final Logger log = LoggerFactory.getLogger(PerformanceTester.class);

  private int batches = 100;
  private int samples = 5;
  private boolean testEvaluation = true;
  private boolean testLearning = true;

  public PerformanceTester() {
  }

  public int getBatches() {
    return batches;
  }

  public void setBatches(int batches) {
    this.batches = batches;
  }

  public int getSamples() {
    return samples;
  }

  public void setSamples(int samples) {
    this.samples = samples;
  }

  public boolean isTestEvaluation() {
    return testEvaluation;
  }

  @Nonnull
  public void setTestEvaluation(final boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  @Nonnull
  public void setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
  }

  public void test(@Nonnull final Layer component, @Nonnull final Tensor[] inputPrototype) {
    log.info(RefString.format("%s batch length, %s trials", batches, samples));
    log.info("Input Dimensions:");
    RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(t -> {
      String temp_10_0001 = "\t" + RefArrays.toString(t.getDimensions());
      t.freeRef();
      return temp_10_0001;
    }).forEach(x1 -> RefSystem.out.println(x1));
    log.info("Performance:");
    RefList<Tuple2<Double, Double>> performance = RefIntStream.range(0, samples)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tuple2<Double, Double>>) i -> {
          return testPerformance(component.addRef(), RefUtil.addRefs(inputPrototype));
        }, component, inputPrototype)).collect(RefCollectors.toList());
    if (isTestEvaluation()) {
      @Nonnull final DoubleStatistics statistics = new DoubleStatistics()
          .accept(performance.stream().mapToDouble(x -> x._1).toArray());
      log.info(RefString.format("\tEvaluation performance: %.6fs +- %.6fs [%.6fs - %.6fs]", statistics.getAverage(),
          statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (isTestLearning()) {
      @Nonnull final DoubleStatistics statistics = new DoubleStatistics()
          .accept(performance.stream().mapToDouble(x -> x._2).toArray());
      log.info(RefString.format("\tLearning performance: %.6fs +- %.6fs [%.6fs - %.6fs]", statistics.getAverage(),
          statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (null != performance)
      performance.freeRef();
  }

  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    log.h1("Performance");
    if (component instanceof DAGNetwork) {
      TestUtil.instrumentPerformance(((DAGNetwork) component).addRef());
    }
    log.p("Now we execute larger-scale runs to benchmark performance:");
    log.run(RefUtil.wrapInterface(() -> {
      test(component == null ? null : component.addRef(), RefUtil.addRefs(inputPrototype));
    }, inputPrototype, component == null ? null : component.addRef()));
    if (component instanceof DAGNetwork) {
      TestUtil.extractPerformance(log, (DAGNetwork) component);
    } else if (null != component) component.freeRef();

    return null;
  }

  @Nonnull
  @Override
  public String toString() {
    return "PerformanceTester{" + "batches=" + batches + ", samples=" + samples + ", testEvaluation=" + testEvaluation
        + ", testLearning=" + testLearning + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PerformanceTester addRef() {
    return (PerformanceTester) super.addRef();
  }

  @Nonnull
  protected Tuple2<Double, Double> testPerformance(@Nonnull final Layer component, @Nullable final Tensor... inputPrototype) {
    final Tensor[][] data = new Tensor[batches][];
    for (int i = 0; i < batches; i++) {
      RefUtil.set(data, i, RefUtil.addRefs(inputPrototype));
    }
    RefUtil.freeRef(inputPrototype);
    long startTime = System.nanoTime();
    final Result result = eval(component, ConstantResult.batchResultArray(data));
    long timeNanos = System.nanoTime() - startTime;
    startTime = System.nanoTime();
    TensorList resultData = result.getData();
    try {
      result.accumulate(new DeltaSet<UUID>(), new TensorArray(resultData.stream().map(x -> {
        try {
          return x.map(v -> 1.0);
        } finally {
          x.freeRef();
        }
      }).toArray(Tensor[]::new)));
    } finally {
      resultData.freeRef();
      result.freeRef();
    }
    long timedBackprop = System.nanoTime() - startTime;
    return new Tuple2<>(timeNanos / 1e9, timedBackprop / 1e9);
  }

  private Result eval(@Nonnull Layer component, Result[] input) {
    try {
      return component.eval(input);
    } finally {
      component.freeRef();
    }
  }
}
