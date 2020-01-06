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

import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.data.DoubleStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Stream;

public @RefAware
class PerformanceTester extends ComponentTestBase<ToleranceStatistics> {
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

  @Nonnull
  public PerformanceTester setBatches(final int batches) {
    this.batches = batches;
    return this.addRef();
  }

  public int getSamples() {
    return samples;
  }

  @Nonnull
  public PerformanceTester setSamples(final int samples) {
    this.samples = samples;
    return this.addRef();
  }

  public boolean isTestEvaluation() {
    return testEvaluation;
  }

  @Nonnull
  public PerformanceTester setTestEvaluation(final boolean testEvaluation) {
    this.testEvaluation = testEvaluation;
    return this.addRef();
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  public static @SuppressWarnings("unused")
  PerformanceTester[] addRefs(PerformanceTester[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PerformanceTester::addRef)
        .toArray((x) -> new PerformanceTester[x]);
  }

  public static @SuppressWarnings("unused")
  PerformanceTester[][] addRefs(PerformanceTester[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PerformanceTester::addRefs)
        .toArray((x) -> new PerformanceTester[x][]);
  }

  @Nonnull
  public ComponentTest<ToleranceStatistics> setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this.addRef();
  }

  public void test(@Nonnull final Layer component, @Nonnull final Tensor[] inputPrototype) {
    log.info(RefString.format("%s batch length, %s trials", batches, samples));
    log.info("Input Dimensions:");
    RefArrays.stream(Tensor.addRefs(inputPrototype)).map(t -> {
      String temp_10_0001 = "\t" + RefArrays.toString(t.getDimensions());
      if (null != t)
        t.freeRef();
      return temp_10_0001;
    }).forEach(com.simiacryptus.ref.wrappers.RefSystem.out::println);
    log.info("Performance:");
    RefList<Tuple2<Double, Double>> performance = RefIntStream.range(0, samples)
        .mapToObj(RefUtil.wrapInterface(
            (IntFunction<? extends Tuple2<Double, Double>>) i -> {
              return testPerformance(component == null ? null : component.addRef(),
                  Tensor.addRefs(inputPrototype));
            }, component == null ? null : component, Tensor.addRefs(inputPrototype)))
        .collect(RefCollectors.toList());
    ReferenceCounting.freeRefs(inputPrototype);
    if (isTestEvaluation()) {
      @Nonnull final DoubleStatistics statistics = new DoubleStatistics()
          .accept(performance.stream().mapToDouble(x -> x._1).toArray());
      log.info(RefString.format("\tEvaluation performance: %.6fs +- %.6fs [%.6fs - %.6fs]", statistics.getAverage(),
          statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
    }
    if (isTestLearning()) {
      @Nonnull final DoubleStatistics statistics = new DoubleStatistics()
          .accept(performance.stream().mapToDouble(x -> x._2).toArray());
      if (null != statistics) {
        log.info(RefString.format("\tLearning performance: %.6fs +- %.6fs [%.6fs - %.6fs]", statistics.getAverage(),
            statistics.getStandardDeviation(), statistics.getMin(), statistics.getMax()));
      }
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
      test(component == null ? null : component.addRef(),
          Tensor.addRefs(inputPrototype));
    }, Tensor.addRefs(inputPrototype), component == null ? null : component.addRef()));
    ReferenceCounting.freeRefs(inputPrototype);
    if (component instanceof DAGNetwork) {
      TestUtil.extractPerformance(log, ((DAGNetwork) component).addRef());
    }
    if (null != component)
      component.freeRef();
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
  }

  public @Override
  @SuppressWarnings("unused")
  PerformanceTester addRef() {
    return (PerformanceTester) super.addRef();
  }

  @Nonnull
  protected Tuple2<Double, Double> testPerformance(@Nonnull final Layer component, final Tensor... inputPrototype) {
    final Tensor[][] data = RefIntStream.range(0, batches).mapToObj(x -> x)
        .flatMap(RefUtil.wrapInterface(
            (Function<? super Integer, ? extends Stream<? extends Tensor[]>>) x -> RefStream
                .<Tensor[]>of(Tensor.addRefs(inputPrototype)),
            Tensor.addRefs(inputPrototype)))
        .toArray(i -> new Tensor[i][]);
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    @Nonnull
    TimedResult<Result> timedEval = TimedResult.time(RefUtil
        .wrapInterface((UncheckedSupplier<Result>) () -> {
          Result[] input = ConstantResult.batchResultArray(Tensor.addRefs(data));
          @Nullable
          Result result;
          try {
            result = component.eval(Result.addRefs(input));
          } finally {
            for (@Nonnull
                Result nnResult : input) {
              RefUtil.freeRef(nnResult.getData());
            }
          }
          if (null != input)
            ReferenceCounting.freeRefs(input);
          return result;
        }, Tensor.addRefs(data), component == null ? null : component));
    if (null != data)
      ReferenceCounting.freeRefs(data);
    final Result result = timedEval.result.addRef();
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    try {
      long timedBackprop = TimedResult.time(RefUtil.wrapInterface(
          (UncheckedSupplier<DeltaSet<UUID>>) () -> {
            TensorList temp_10_0003 = result.getData();
            @Nonnull
            TensorArray tensorArray = new TensorArray(temp_10_0003.stream().map(x -> {
              Tensor temp_10_0002 = x.map(v -> 1.0);
              if (null != x)
                x.freeRef();
              return temp_10_0002;
            }).toArray(i -> new Tensor[i]));
            if (null != temp_10_0003)
              temp_10_0003.freeRef();
            result.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
            return buffer;
          }, buffer == null ? null : buffer, result == null ? null : result.addRef())).timeNanos;
      if (null != result)
        result.freeRef();
      return new Tuple2<>(timedEval.timeNanos / 1e9, timedBackprop / 1e9);
    } finally {
      RefUtil.freeRef(result.getData());
    }
  }
}
