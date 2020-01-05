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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.PlaceholderLayer;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.data.ScalarStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.IntFunction;

public @RefAware
class BatchDerivativeTester extends ComponentTestBase<ToleranceStatistics> {
  static final Logger log = LoggerFactory.getLogger(BatchDerivativeTester.class);

  public final double probeSize;
  private final int batches;
  private final double tolerance;
  private boolean testFeedback = true;
  private boolean testLearning = true;
  private boolean verbose = true;
  private boolean verify = true;

  public BatchDerivativeTester(final double tolerance, final double probeSize, final int batches) {
    this.tolerance = tolerance;
    this.probeSize = probeSize;
    this.batches = batches;
  }

  public boolean isTestFeedback() {
    return testFeedback;
  }

  @Nonnull
  public BatchDerivativeTester setTestFeedback(final boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this.addRef();
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  @Nonnull
  public BatchDerivativeTester setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this.addRef();
  }

  public boolean isVerbose() {
    return verbose;
  }

  @Nonnull
  public BatchDerivativeTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this.addRef();
  }

  public boolean isVerify() {
    return verify;
  }

  @Nonnull
  public BatchDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this.addRef();
  }

  public static @SuppressWarnings("unused")
  BatchDerivativeTester[] addRefs(BatchDerivativeTester[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BatchDerivativeTester::addRef)
        .toArray((x) -> new BatchDerivativeTester[x]);
  }

  public static @SuppressWarnings("unused")
  BatchDerivativeTester[][] addRefs(BatchDerivativeTester[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BatchDerivativeTester::addRefs)
        .toArray((x) -> new BatchDerivativeTester[x][]);
  }

  public ToleranceStatistics testLearning(@Nonnull Layer component, @Nonnull IOPair IOPair,
                                          ToleranceStatistics statistics) {
    final ToleranceStatistics prev = statistics;
    RefList<double[]> temp_02_0021 = component.state();
    statistics = RefIntStream.range(0, temp_02_0021.size()).mapToObj(RefUtil.wrapInterface(
        (IntFunction<? extends ToleranceStatistics>) i -> {
          Tensor temp_02_0022 = measureLearningGradient(
              component == null ? null : component.addRef(), i, IOPair.getOutputPrototype(),
              IOPair.getInputPrototype());
          @Nullable final Tensor measuredGradient = !verify ? null : temp_02_0022.addRef();
          if (null != temp_02_0022)
            temp_02_0022.freeRef();
          @Nonnull final Tensor implementedGradient = getLearningGradient(component == null ? null : component.addRef(), i,
              IOPair.getOutputPrototype(), IOPair.getInputPrototype());
          try {
            final ToleranceStatistics result = RefIntStream
                .range(0, null == measuredGradient ? 0 : measuredGradient.length())
                .mapToObj(RefUtil.wrapInterface(
                    (IntFunction<? extends ToleranceStatistics>) i1 -> {
                      return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                          implementedGradient.getData()[i1]);
                    }, measuredGradient == null ? null : measuredGradient.addRef(),
                    implementedGradient == null ? null : implementedGradient.addRef()))
                .reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
            if (!(result.absoluteTol.getMax() < tolerance)) {
              if (null != measuredGradient)
                measuredGradient.freeRef();
              implementedGradient.freeRef();
              throw new AssertionError(result.toString());
            } else {
              //log.info(String.format("Component: %s", component));
              if (verbose) {

                log.info(String.format("Learning Gradient for weight setByCoord %s", i));
                RefList<double[]> temp_02_0023 = component.state();
                Tensor temp_02_0018 = new Tensor(temp_02_0023.get(i));
                if (null != temp_02_0023)
                  temp_02_0023.freeRef();
                log.info(String.format("Weights: %s", temp_02_0018.prettyPrint()));
                if (null != temp_02_0018)
                  temp_02_0018.freeRef();
                log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
                log.info(String.format("Implemented Statistics: %s",
                    new ScalarStatistics().add(implementedGradient.getData())));
                if (null != measuredGradient) {
                  log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
                  log.info(
                      String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                  Tensor temp_02_0024 = measuredGradient
                      .minus(implementedGradient == null ? null : implementedGradient.addRef());
                  log.info(String.format("Gradient Error: %s", temp_02_0024.prettyPrint()));
                  if (null != temp_02_0024)
                    temp_02_0024.freeRef();
                  Tensor temp_02_0025 = measuredGradient
                      .minus(implementedGradient == null ? null : implementedGradient.addRef());
                  log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(temp_02_0025.getData())));
                  if (null != temp_02_0025)
                    temp_02_0025.freeRef();
                }
              }
              if (null != measuredGradient)
                measuredGradient.freeRef();
              implementedGradient.freeRef();
              return result;
            }
          } catch (@Nonnull final Throwable e) {
            //log.info(String.format("Component: %s", component));
            log.info(String.format("Learning Gradient for weight setByCoord %s", i));
            log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
            log.info(
                String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
              log.info(
                  String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              Tensor temp_02_0026 = measuredGradient
                  .minus(implementedGradient == null ? null : implementedGradient.addRef());
              log.info(String.format("Gradient Error: %s", temp_02_0026.prettyPrint()));
              if (null != temp_02_0026)
                temp_02_0026.freeRef();
              Tensor temp_02_0027 = measuredGradient
                  .minus(implementedGradient == null ? null : implementedGradient.addRef());
              log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(temp_02_0027.getData())));
              if (null != temp_02_0027)
                temp_02_0027.freeRef();
            }
            throw e;
          }

        }, component == null ? null : component, IOPair == null ? null : IOPair)).reduce((a, b) -> a.combine(b))
        .map(x -> x.combine(prev)).orElse(prev);
    if (null != temp_02_0021)
      temp_02_0021.freeRef();
    return statistics;
  }

  public ToleranceStatistics testFeedback(@Nonnull Layer component, @Nonnull IOPair IOPair,
                                          ToleranceStatistics statistics) {
    Tensor[] temp_02_0028 = IOPair.getInputPrototype();
    statistics = statistics
        .combine(RefIntStream.range(0, temp_02_0028.length).mapToObj(RefUtil.wrapInterface(
            (IntFunction<? extends ToleranceStatistics>) i -> {
              Tensor temp_02_0029 = measureFeedbackGradient(
                  component == null ? null : component.addRef(), i, IOPair.getOutputPrototype(),
                  IOPair.getInputPrototype());
              @Nullable final Tensor measuredGradient = !verify ? null : temp_02_0029.addRef();
              if (null != temp_02_0029)
                temp_02_0029.freeRef();
              @Nonnull final Tensor implementedGradient = getFeedbackGradient(component == null ? null : component.addRef(), i,
                  IOPair.getOutputPrototype(), IOPair.getInputPrototype());
              try {
                final ToleranceStatistics result = RefIntStream
                    .range(0, null == measuredGradient ? 0 : measuredGradient.length())
                    .mapToObj(RefUtil.wrapInterface(
                        (IntFunction<? extends ToleranceStatistics>) i1 -> {
                          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                              implementedGradient.getData()[i1]);
                        }, implementedGradient == null ? null : implementedGradient.addRef(),
                        measuredGradient == null ? null : measuredGradient.addRef()))
                    .reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());

                if (!(result.absoluteTol.getMax() < tolerance)) {
                  if (null != measuredGradient)
                    measuredGradient.freeRef();
                  implementedGradient.freeRef();
                  throw new AssertionError(result.toString());
                }
                //log.info(String.format("Component: %s", component));
                if (verbose) {
                  log.info(String.format("Feedback for input %s", i));
                  log.info(String.format("Inputs Values: %s", IOPair.getInputPrototype()[i].prettyPrint()));
                  log.info(String.format("Value Statistics: %s",
                      new ScalarStatistics().add(IOPair.getInputPrototype()[i].getData())));
                  log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
                  log.info(String.format("Implemented Statistics: %s",
                      new ScalarStatistics().add(implementedGradient.getData())));
                  if (null != measuredGradient) {
                    log.info(String.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
                    log.info(String.format("Measured Statistics: %s",
                        new ScalarStatistics().add(measuredGradient.getData())));
                    Tensor temp_02_0030 = measuredGradient
                        .minus(implementedGradient == null ? null : implementedGradient.addRef());
                    log.info(String.format("Feedback Error: %s", temp_02_0030.prettyPrint()));
                    if (null != temp_02_0030)
                      temp_02_0030.freeRef();
                    Tensor temp_02_0031 = measuredGradient
                        .minus(implementedGradient == null ? null : implementedGradient.addRef());
                    log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(temp_02_0031.getData())));
                    if (null != temp_02_0031)
                      temp_02_0031.freeRef();
                  }
                }
                if (null != measuredGradient)
                  measuredGradient.freeRef();
                implementedGradient.freeRef();
                return result;
              } catch (@Nonnull final Throwable e) {
                //log.info(String.format("Component: %s", component));
                log.info(String.format("Feedback for input %s", i));
                log.info(String.format("Inputs Values: %s", IOPair.getInputPrototype()[i].prettyPrint()));
                log.info(String.format("Value Statistics: %s",
                    new ScalarStatistics().add(IOPair.getInputPrototype()[i].getData())));
                log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
                log.info(String.format("Implemented Statistics: %s",
                    new ScalarStatistics().add(implementedGradient.getData())));
                if (null != measuredGradient) {
                  log.info(String.format("Measured: %s", measuredGradient.prettyPrint()));
                  log.info(
                      String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                  Tensor temp_02_0032 = measuredGradient
                      .minus(implementedGradient == null ? null : implementedGradient.addRef());
                  log.info(String.format("Feedback Error: %s", temp_02_0032.prettyPrint()));
                  if (null != temp_02_0032)
                    temp_02_0032.freeRef();
                  Tensor temp_02_0033 = measuredGradient
                      .minus(implementedGradient == null ? null : implementedGradient.addRef());
                  log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(temp_02_0033.getData())));
                  if (null != temp_02_0033)
                    temp_02_0033.freeRef();
                }
                throw e;
              }
            }, IOPair == null ? null : IOPair, component == null ? null : component)).reduce((a, b) -> a.combine(b))
            .get());
    if (null != temp_02_0028)
      ReferenceCounting.freeRefs(temp_02_0028);
    return statistics;
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    log.h1("Differential Validation");
    BatchDerivativeTester.IOPair temp_02_0019 = new IOPair(
        component == null ? null : component.addRef(), inputPrototype[0].addRef(), BatchDerivativeTester.this.addRef());
    @Nonnull
    IOPair ioPair = temp_02_0019.invoke();

    if (null != temp_02_0019)
      temp_02_0019.freeRef();
    if (verbose) {
      log.run(RefUtil.wrapInterface(() -> {
        BatchDerivativeTester.log.info(String.format("Inputs: %s",
            RefArrays.stream(Tensor.addRefs(inputPrototype)).map(t -> {
              String temp_02_0007 = t.prettyPrint();
              if (null != t)
                t.freeRef();
              return temp_02_0007;
            }).reduce((a, b) -> a + ",\n" + b).get()));
        BatchDerivativeTester.log.info(String.format("Inputs Statistics: %s",
            RefArrays.stream(Tensor.addRefs(inputPrototype)).map(x -> {
              String temp_02_0008 = new ScalarStatistics().add(x.getData()).toString();
              if (null != x)
                x.freeRef();
              return temp_02_0008;
            }).reduce((a, b) -> a + ",\n" + b).get()));
        Tensor temp_02_0034 = ioPair.getOutputPrototype();
        BatchDerivativeTester.log.info(String.format("Output: %s", temp_02_0034.prettyPrint()));
        if (null != temp_02_0034)
          temp_02_0034.freeRef();
        Tensor temp_02_0035 = ioPair.getOutputPrototype();
        BatchDerivativeTester.log
            .info(String.format("Outputs Statistics: %s", new ScalarStatistics().add(temp_02_0035.getData())));
        if (null != temp_02_0035)
          temp_02_0035.freeRef();
      }, Tensor.addRefs(inputPrototype), ioPair == null ? null : ioPair.addRef()));
    }

    ReferenceCounting.freeRefs(inputPrototype);
    ToleranceStatistics _statistics = new ToleranceStatistics();

    if (isTestFeedback()) {
      log.h2("Feedback Validation");
      log.p(
          "We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.eval(RefUtil.wrapInterface(
          () -> {
            return testFeedback(component == null ? null : component.addRef(), ioPair == null ? null : ioPair.addRef(),
                statistics);
          }, component == null ? null : component.addRef(), ioPair == null ? null : ioPair.addRef()));
    }
    if (isTestLearning()) {
      log.h2("Learning Validation");
      log.p(
          "We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.eval(RefUtil.wrapInterface(
          () -> {
            return testLearning(component == null ? null : component.addRef(), ioPair == null ? null : ioPair.addRef(),
                statistics);
          }, component == null ? null : component.addRef(), ioPair == null ? null : ioPair.addRef()));
    }

    log.h2("Total Accuracy");
    log.p("The overall agreement accuracy between the implemented derivative and the finite difference estimations:");
    ToleranceStatistics statistics = _statistics;
    log.run(() -> {
      //log.info(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toStream(inputPrototype), outputPrototype));
      BatchDerivativeTester.log.info(String.format("Finite-Difference Derivative Accuracy:"));
      BatchDerivativeTester.log.info(String.format("absoluteTol: %s", statistics.absoluteTol));
      BatchDerivativeTester.log.info(String.format("relativeTol: %s", statistics.relativeTol));
    });

    log.h2("Frozen and Alive Status");
    log.run(RefUtil.wrapInterface(() -> {
      testFrozen(component == null ? null : component.addRef(), ioPair.getInputPrototype());
      testUnFrozen(component == null ? null : component.addRef(), ioPair.getInputPrototype());
    }, component == null ? null : component, ioPair == null ? null : ioPair));

    return _statistics;
  }

  public void testFrozen(@Nonnull final Layer component, @Nonnull final Tensor[] inputPrototype) {
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_02_0036 = component.copy();
    @Nonnull final Layer frozen = temp_02_0036.freeze();
    if (null != temp_02_0036)
      temp_02_0036.freeRef();
    @Nullable final Result eval = frozen.eval(new Result(
        new TensorArray(Tensor.addRefs(inputPrototype)), new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList data) {
        reachedInputFeedback.set(true);
      }
    }) {

      @Override
      public boolean isAlive() {
        return true;
      }

      public @SuppressWarnings("unused")
      void _free() {
      }

    });
    frozen.freeRef();
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList temp_02_0037 = eval.getData();
    TensorList tensorList = temp_02_0037.copy();
    if (null != temp_02_0037)
      temp_02_0037.freeRef();
    eval.accumulate(buffer == null ? null : buffer.addRef(), tensorList == null ? null : tensorList.addRef());
    if (null != tensorList)
      tensorList.freeRef();
    if (null != eval)
      eval.freeRef();
    RefList<double[]> temp_02_0038 = component.state();
    final RefList<Delta<UUID>> deltas = temp_02_0038.stream().map(RefUtil.wrapInterface(
        (Function<? super double[], ? extends Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_02_0040 = buffer.stream()
              .filter(x -> {
                boolean temp_02_0009 = x.target == doubles;
                if (null != x)
                  x.freeRef();
                return temp_02_0009;
              }).findFirst();
          Delta<UUID> temp_02_0039 = temp_02_0040.orElse(null);
          if (null != temp_02_0040)
            RefUtil.freeRef(temp_02_0040);
          return temp_02_0039;
        }, buffer == null ? null : buffer)).filter(x -> {
      boolean temp_02_0010 = x != null;
      if (null != x)
        x.freeRef();
      return temp_02_0010;
    }).collect(RefCollectors.toList());
    if (null != temp_02_0038)
      temp_02_0038.freeRef();
    RefList<double[]> temp_02_0041 = component.state();
    if (!deltas.isEmpty() && !temp_02_0041.isEmpty()) {
      AssertionError temp_02_0011 = new AssertionError(
          "Frozen component listed in evalInputDelta. Deltas: " + deltas);
      if (null != deltas)
        deltas.freeRef();
      component.freeRef();
      ReferenceCounting.freeRefs(inputPrototype);
      throw temp_02_0011;
    }
    if (null != temp_02_0041)
      temp_02_0041.freeRef();
    component.freeRef();
    if (null != deltas)
      deltas.freeRef();
    final int inElements = RefArrays.stream(Tensor.addRefs(inputPrototype))
        .mapToInt(x -> {
          int temp_02_0012 = x.length();
          if (null != x)
            x.freeRef();
          return temp_02_0012;
        }).sum();
    ReferenceCounting.freeRefs(inputPrototype);
    if (!reachedInputFeedback.get() && 0 < inElements) {
      throw new RuntimeException("Frozen component did not pass input backwards");
    }
  }

  public void testUnFrozen(@Nonnull final Layer component, final Tensor[] inputPrototype) {
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_02_0042 = component.copy();
    @Nonnull final Layer frozen = temp_02_0042.setFrozen(false);
    if (null != temp_02_0042)
      temp_02_0042.freeRef();
    component.freeRef();
    @Nullable final Result eval = frozen.eval(new Result(
        new TensorArray(Tensor.addRefs(inputPrototype)), new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList data) {
        reachedInputFeedback.set(true);
      }
    }) {

      @Override
      public boolean isAlive() {
        return true;
      }

      public @SuppressWarnings("unused")
      void _free() {
      }

    });
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList data = eval.getData();
    eval.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data.addRef());
    if (null != data)
      data.freeRef();
    if (null != eval)
      eval.freeRef();
    @Nullable final RefList<double[]> stateList = frozen.state();
    frozen.freeRef();
    final RefList<Delta<UUID>> deltas = stateList.stream().map(RefUtil.wrapInterface(
        (Function<? super double[], ? extends Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_02_0044 = buffer.stream()
              .filter(x -> {
                boolean temp_02_0013 = x.target == doubles;
                if (null != x)
                  x.freeRef();
                return temp_02_0013;
              }).findFirst();
          Delta<UUID> temp_02_0043 = temp_02_0044.orElse(null);
          if (null != temp_02_0044)
            RefUtil.freeRef(temp_02_0044);
          return temp_02_0043;
        }, buffer == null ? null : buffer)).filter(x -> {
      boolean temp_02_0014 = x != null;
      if (null != x)
        x.freeRef();
      return temp_02_0014;
    }).collect(RefCollectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      if (null != stateList)
        stateList.freeRef();
      AssertionError temp_02_0015 = new AssertionError(
          "Nonfrozen component not listed in evalInputDelta. Deltas: " + deltas);
      if (null != deltas)
        deltas.freeRef();
      throw temp_02_0015;
    }
    if (null != deltas)
      deltas.freeRef();
    if (null != stateList)
      stateList.freeRef();
    if (!reachedInputFeedback.get()) {
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
  }

  @Nonnull
  @Override
  public String toString() {
    return "BatchDerivativeTester{" + "probeSize=" + probeSize + ", batches=" + batches + ", tolerance=" + tolerance
        + ", testFeedback=" + testFeedback + ", testLearning=" + testLearning + ", verbose=" + verbose + ", verify="
        + verify + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  BatchDerivativeTester addRef() {
    return (BatchDerivativeTester) super.addRef();
  }

  @Nonnull
  private Tensor getFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                     @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex].addRef();
    final int inputDims = inputTensor.length();
    @Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor());
      @Nonnull final Result copyInput = new Result(
          new TensorArray(Tensor.addRefs(inputPrototype)), new Result.Accumulator() {
        @Override
        public void accept(DeltaSet<UUID> buffer, TensorList data) {
          @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
          if (!RefArrays.equals(inputTensor.getDimensions(), data.get(inputIndex).getDimensions())) {
            throw new AssertionError();
          }
          for (int i = 0; i < inputDims; i++) {
            gradientBuffer.set(new int[]{i, j_}, data.get(inputIndex).getData()[i]);
          }
          buffer.get(inputKey.getId(), new double[gradientBuffer.length()]).addInPlace(gradientBuffer.getData());
        }
      }) {

        @Override
        public boolean isAlive() {
          return true;
        }

        public @SuppressWarnings("unused")
        void _free() {
        }

      };
      @Nullable final Result eval = component.eval(copyInput == null ? null : copyInput);
      @Nonnull final DeltaSet<UUID> xxx = new DeltaSet<UUID>();
      TensorList temp_02_0045 = eval.getData();
      @Nonnull
      TensorArray tensorArray = new TensorArray(temp_02_0045.stream().map(x -> {
        Tensor temp_02_0016 = x.set(j_, 1);
        if (null != x)
          x.freeRef();
        return temp_02_0016;
      }).toArray(i -> new Tensor[i]));
      if (null != temp_02_0045)
        temp_02_0045.freeRef();
      eval.accumulate(xxx == null ? null : xxx.addRef(), tensorArray == null ? null : tensorArray);
      if (null != eval)
        eval.freeRef();
      RefMap<UUID, Delta<UUID>> temp_02_0046 = xxx
          .getMap();
      final Delta<UUID> inputDelta = temp_02_0046.get(inputKey == null ? null : inputKey);
      if (null != temp_02_0046)
        temp_02_0046.freeRef();
      xxx.freeRef();
      if (null != inputDelta) {
        result.addInPlace(new Tensor(inputDelta.getDelta(), result.getDimensions()));
      }
      if (null != inputDelta)
        inputDelta.freeRef();
    }
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    outputPrototype.freeRef();
    component.freeRef();
    if (null != inputTensor)
      inputTensor.freeRef();
    return result;
  }

  @Nonnull
  private Tensor getLearningGradient(@Nonnull final Layer component, final int layerNum,
                                     @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    RefUtil.freeRef(component.setFrozen(false));
    RefList<double[]> temp_02_0047 = component.state();
    final double[] stateArray = temp_02_0047.get(layerNum);
    if (null != temp_02_0047)
      temp_02_0047.freeRef();
    final int stateLen = stateArray.length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
      Tensor temp_02_0020 = new Tensor(outputPrototype.getDimensions());
      @Nonnull final Tensor data = temp_02_0020.set((k) -> k == j_ ? 1 : 0);
      if (null != temp_02_0020)
        temp_02_0020.freeRef();
      @Nullable final Result eval = component.eval(ConstantResult
          .singleResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)}));
      TensorList temp_02_0048 = eval.getData();
      RefUtil.freeRef(temp_02_0048.get(0));
      if (null != temp_02_0048)
        temp_02_0048.freeRef();
      @Nonnull
      TensorArray tensorArray = new TensorArray(data == null ? null : data);
      eval.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
      if (null != eval)
        eval.freeRef();
      RefMap<UUID, Delta<UUID>> temp_02_0049 = buffer
          .getMap();
      RefCollection<Delta<UUID>> temp_02_0050 = temp_02_0049
          .values();
      Optional<Delta<UUID>> temp_02_0051 = temp_02_0050.stream()
          .filter(x -> {
            boolean temp_02_0017 = x.target == stateArray;
            if (null != x)
              x.freeRef();
            return temp_02_0017;
          }).findFirst();
      final DoubleBuffer<UUID> deltaFlushBuffer = temp_02_0051.orElse(null);
      if (null != temp_02_0051)
        RefUtil.freeRef(temp_02_0051);
      if (null != temp_02_0050)
        temp_02_0050.freeRef();
      if (null != temp_02_0049)
        temp_02_0049.freeRef();
      buffer.freeRef();
      if (null != deltaFlushBuffer) {
        for (int i = 0; i < stateLen; i++) {
          gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
        }
      }
      if (null != deltaFlushBuffer)
        deltaFlushBuffer.freeRef();
    }
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    outputPrototype.freeRef();
    component.freeRef();
    return gradient;
  }

  @Nonnull
  private Tensor measureFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                         @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    @Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].length(), outputPrototype.length());
    Result temp_02_0052 = component.eval(ConstantResult
        .singleResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)}));
    TensorList temp_02_0053 = temp_02_0052.getData();
    @Nullable final Tensor baseOutput = temp_02_0053.get(0);
    if (null != temp_02_0053)
      temp_02_0053.freeRef();
    if (null != temp_02_0052)
      temp_02_0052.freeRef();
    outputPrototype.set(baseOutput == null ? null : baseOutput.addRef());
    outputPrototype.freeRef();
    for (int i = 0; i < inputPrototype[inputIndex].length(); i++) {
      @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      @Nonnull final Tensor[] copyInput = RefArrays.copyOf(Tensor.addRefs(inputPrototype),
          inputPrototype.length);
      {
        Tensor temp_02_0001 = inputProbe == null ? null : inputProbe.addRef();
        if (null != copyInput[inputIndex])
          copyInput[inputIndex].freeRef();
        copyInput[inputIndex] = temp_02_0001 == null ? null : temp_02_0001.addRef();
        if (null != temp_02_0001)
          temp_02_0001.freeRef();
      }
      inputProbe.freeRef();
      Result temp_02_0054 = component.eval(ConstantResult
          .singleResultArray(new Tensor[][]{Tensor.addRefs(copyInput)}));
      TensorList temp_02_0055 = temp_02_0054.getData();
      @Nullable final Tensor evalProbe = temp_02_0055.get(0);
      if (null != temp_02_0055)
        temp_02_0055.freeRef();
      if (null != temp_02_0054)
        temp_02_0054.freeRef();
      ReferenceCounting.freeRefs(copyInput);
      Tensor temp_02_0056 = evalProbe
          .minus(baseOutput == null ? null : baseOutput.addRef());
      @Nonnull final Tensor delta = temp_02_0056.scaleInPlace(1. / probeSize);
      if (null != temp_02_0056)
        temp_02_0056.freeRef();
      if (null != evalProbe)
        evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    ReferenceCounting.freeRefs(inputPrototype);
    component.freeRef();
    if (null != baseOutput)
      baseOutput.freeRef();
    return measuredGradient;
  }

  @Nonnull
  private Tensor measureLearningGradient(@Nonnull final Layer component, final int layerNum,
                                         @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    RefList<double[]> temp_02_0057 = component.state();
    final int stateLen = temp_02_0057.get(layerNum).length;
    if (null != temp_02_0057)
      temp_02_0057.freeRef();
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());

    outputPrototype.freeRef();
    Result temp_02_0058 = component.eval(ConstantResult
        .singleResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)}));
    TensorList temp_02_0059 = temp_02_0058.getData();
    @Nullable final Tensor baseOutput = temp_02_0059.get(0);

    if (null != temp_02_0059)
      temp_02_0059.freeRef();
    if (null != temp_02_0058)
      temp_02_0058.freeRef();
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      RefList<double[]> temp_02_0060 = copy.state();
      temp_02_0060.get(layerNum)[i] += probeSize;

      if (null != temp_02_0060)
        temp_02_0060.freeRef();
      Result temp_02_0061 = copy.eval(ConstantResult
          .singleResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)}));
      TensorList temp_02_0062 = temp_02_0061.getData();
      @Nullable final Tensor evalProbe = temp_02_0062.get(0);

      if (null != temp_02_0062)
        temp_02_0062.freeRef();
      if (null != temp_02_0061)
        temp_02_0061.freeRef();
      copy.freeRef();
      Tensor temp_02_0063 = evalProbe
          .minus(baseOutput == null ? null : baseOutput.addRef());
      @Nonnull final Tensor delta = temp_02_0063.scaleInPlace(1. / probeSize);
      if (null != temp_02_0063)
        temp_02_0063.freeRef();
      if (null != evalProbe)
        evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    component.freeRef();
    if (null != baseOutput)
      baseOutput.freeRef();
    return gradient;
  }

  private static @RefAware
  class IOPair extends ReferenceCountingBase {
    private final Layer component;
    private final Tensor tensor;
    private final BatchDerivativeTester parent;
    private Tensor[] inputPrototype;
    private Tensor outputPrototype;

    public IOPair(Layer component, Tensor tensor, BatchDerivativeTester parent) {
      {
        Layer temp_02_0002 = component == null ? null : component.addRef();
        this.component = temp_02_0002 == null ? null : temp_02_0002.addRef();
        if (null != temp_02_0002)
          temp_02_0002.freeRef();
      }
      if (null != component)
        component.freeRef();
      {
        Tensor temp_02_0003 = tensor == null ? null : tensor.addRef();
        this.tensor = temp_02_0003 == null ? null : temp_02_0003.addRef();
        if (null != temp_02_0003)
          temp_02_0003.freeRef();
      }
      if (null != tensor)
        tensor.freeRef();
      {
        BatchDerivativeTester temp_02_0004 = parent == null ? null
            : parent.addRef();
        this.parent = temp_02_0004 == null ? null : temp_02_0004.addRef();
        if (null != temp_02_0004)
          temp_02_0004.freeRef();
      }
      if (null != parent)
        parent.freeRef();
    }

    public Tensor[] getInputPrototype() {
      return Tensor.addRefs(inputPrototype);
    }

    public Tensor getOutputPrototype() {
      return outputPrototype == null ? null : outputPrototype.addRef();
    }

    public static @SuppressWarnings("unused")
    IOPair[] addRefs(IOPair[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(IOPair::addRef).toArray((x) -> new IOPair[x]);
    }

    @Nonnull
    public IOPair invoke() {
      {
        Tensor[] temp_02_0005 = RefIntStream.range(0, parent.batches)
            .mapToObj(i -> tensor.copy()).toArray(j -> new Tensor[j]);
        if (null != inputPrototype)
          ReferenceCounting.freeRefs(inputPrototype);
        inputPrototype = Tensor.addRefs(temp_02_0005);
        if (null != temp_02_0005)
          ReferenceCounting.freeRefs(temp_02_0005);
      }
      {
        SimpleEval temp_02_0064 = SimpleEval
            .run(component == null ? null : component.addRef(), inputPrototype[0].addRef());
        Tensor temp_02_0006 = temp_02_0064.getOutput();
        if (null != temp_02_0064)
          temp_02_0064.freeRef();
        if (null != outputPrototype)
          outputPrototype.freeRef();
        outputPrototype = temp_02_0006 == null ? null : temp_02_0006.addRef();
        if (null != temp_02_0006)
          temp_02_0006.freeRef();
      }
      return this.addRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      if (null != outputPrototype)
        outputPrototype.freeRef();
      outputPrototype = null;
      if (null != inputPrototype)
        ReferenceCounting.freeRefs(inputPrototype);
      inputPrototype = null;
      if (null != parent)
        parent.freeRef();
      if (null != tensor)
        tensor.freeRef();
      if (null != component)
        component.freeRef();
    }

    public @Override
    @SuppressWarnings("unused")
    IOPair addRef() {
      return (IOPair) super.addRef();
    }
  }
}
