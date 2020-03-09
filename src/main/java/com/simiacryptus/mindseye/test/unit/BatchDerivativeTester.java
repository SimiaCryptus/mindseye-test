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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.IntFunction;

public class BatchDerivativeTester extends ComponentTestBase<ToleranceStatistics> {
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

  public void setTestFeedback(boolean testFeedback) {
    this.testFeedback = testFeedback;
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  public void setTestLearning(boolean testLearning) {
    this.testLearning = testLearning;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  public boolean isVerify() {
    return verify;
  }

  public void setVerify(boolean verify) {
    this.verify = verify;
  }

  public ToleranceStatistics testLearning(@Nonnull Layer component, @Nonnull IOPair IOPair,
                                          ToleranceStatistics statistics) {
    final ToleranceStatistics prev = statistics;
    RefList<double[]> temp_02_0021 = component.state();
    assert temp_02_0021 != null;
    statistics = RefIntStream.range(0, temp_02_0021.size())
        .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i -> {
          Tensor temp_02_0022 = measureLearningGradient(component.addRef(), i,
              IOPair.getOutputPrototype(), IOPair.getInputPrototype());
          @Nullable final Tensor measuredGradient = !verify ? null : temp_02_0022.addRef();
          temp_02_0022.freeRef();
          @Nonnull final Tensor implementedGradient = getLearningGradient(component.addRef(), i,
              IOPair.getOutputPrototype(), IOPair.getInputPrototype());
          try {
            final ToleranceStatistics result = RefIntStream
                .range(0, null == measuredGradient ? 0 : measuredGradient.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i1 -> {
                      assert measuredGradient != null;
                      return new ToleranceStatistics().accumulate(measuredGradient.get(i1),
                          implementedGradient.get(i1));
                    }, measuredGradient == null ? null : measuredGradient.addRef(),
                    implementedGradient.addRef()))
                .reduce(ToleranceStatistics::combine).orElse(new ToleranceStatistics());
            if (!(result.absoluteTol.getMax() < tolerance)) {
              if (null != measuredGradient)
                measuredGradient.freeRef();
              implementedGradient.freeRef();
              throw new AssertionError(result.toString());
            } else {
              //log.info(String.format("Component: %s", component));
              if (verbose) {

                log.info(RefString.format("Learning Gradient for weight setByCoord %s", i));
                RefList<double[]> temp_02_0023 = component.state();
                assert temp_02_0023 != null;
                Tensor temp_02_0018 = new Tensor(temp_02_0023.get(i));
                temp_02_0023.freeRef();
                log.info(RefString.format("Weights: %s", temp_02_0018.prettyPrint()));
                temp_02_0018.freeRef();
                log.info(RefString.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
                log.info(RefString.format("Implemented Statistics: %s",
                    implementedGradient.getScalarStatistics()));
                if (null != measuredGradient) {
                  log.info(RefString.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
                  log.info(RefString.format("Measured Statistics: %s",
                      measuredGradient.getScalarStatistics()));
                  Tensor temp_02_0024 = measuredGradient
                      .minus(implementedGradient.addRef());
                  log.info(RefString.format("Gradient Error: %s", temp_02_0024.prettyPrint()));
                  temp_02_0024.freeRef();
                  Tensor temp_02_0025 = measuredGradient
                      .minus(implementedGradient.addRef());
                  log.info(
                      RefString.format("Error Statistics: %s", temp_02_0025.getScalarStatistics()));
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
            log.info(RefString.format("Learning Gradient for weight setByCoord %s", i));
            log.info(RefString.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
            log.info(RefString.format("Implemented Statistics: %s",
                implementedGradient.getScalarStatistics()));
            if (null != measuredGradient) {
              log.info(RefString.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
              log.info(
                  RefString.format("Measured Statistics: %s", measuredGradient.getScalarStatistics()));
              Tensor temp_02_0026 = measuredGradient
                  .minus(implementedGradient.addRef());
              log.info(RefString.format("Gradient Error: %s", temp_02_0026.prettyPrint()));
              temp_02_0026.freeRef();
              Tensor temp_02_0027 = measuredGradient
                  .minus(implementedGradient.addRef());
              log.info(RefString.format("Error Statistics: %s", temp_02_0027.getScalarStatistics()));
              temp_02_0027.freeRef();
            }
            throw e;
          }
        }, component, IOPair)).reduce(ToleranceStatistics::combine)
        .map(x -> x.combine(prev)).orElse(prev);
    temp_02_0021.freeRef();
    return statistics;
  }

  public ToleranceStatistics testFeedback(@Nonnull Layer component, @Nonnull IOPair IOPair,
                                          ToleranceStatistics statistics) {
    Tensor[] inputPrototype = IOPair.getInputPrototype();
    assert inputPrototype != null;
    statistics = statistics.combine(RefUtil.get(RefIntStream.range(0, inputPrototype.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i -> {
          Tensor temp_02_0029 = measureFeedbackGradient(component.addRef(), i,
              IOPair.getOutputPrototype(), RefUtil.addRef(inputPrototype));
          @Nullable final Tensor measuredGradient = !verify ? null : temp_02_0029.addRef();
          temp_02_0029.freeRef();
          @Nonnull final Tensor implementedGradient = getFeedbackGradient(component.addRef(), i,
              IOPair.getOutputPrototype(), RefUtil.addRef(inputPrototype));
          try {
            final ToleranceStatistics result = RefIntStream
                .range(0, null == measuredGradient ? 0 : measuredGradient.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i1 -> {
                      assert measuredGradient != null;
                      return new ToleranceStatistics().accumulate(measuredGradient.get(i1),
                          implementedGradient.get(i1));
                    }, implementedGradient.addRef(),
                    measuredGradient == null ? null : measuredGradient.addRef()))
                .reduce(ToleranceStatistics::combine).orElse(new ToleranceStatistics());

            if (!(result.absoluteTol.getMax() < tolerance)) {
              if (null != measuredGradient)
                measuredGradient.freeRef();
              implementedGradient.freeRef();
              throw new AssertionError(result.toString());
            }
            //log.info(String.format("Component: %s", component));
            if (verbose) {
              log.info(RefString.format("Feedback for input %s", i));
              log.info(RefString.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
              log.info(RefString.format("Value Statistics: %s",
                  inputPrototype[i].getScalarStatistics()));
              log.info(RefString.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
              log.info(RefString.format("Implemented Statistics: %s",
                  implementedGradient.getScalarStatistics()));
              if (null != measuredGradient) {
                log.info(RefString.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
                log.info(RefString.format("Measured Statistics: %s",
                    measuredGradient.getScalarStatistics()));
                Tensor temp_02_0030 = measuredGradient
                    .minus(implementedGradient.addRef());
                log.info(RefString.format("Feedback Error: %s", temp_02_0030.prettyPrint()));
                temp_02_0030.freeRef();
                Tensor temp_02_0031 = measuredGradient
                    .minus(implementedGradient.addRef());
                log.info(RefString.format("Error Statistics: %s", temp_02_0031.getScalarStatistics()));
                temp_02_0031.freeRef();
              }
            }
            if (null != measuredGradient)
              measuredGradient.freeRef();
            implementedGradient.freeRef();
            return result;
          } catch (@Nonnull final Throwable e) {
            //log.info(String.format("Component: %s", component));
            log.info(RefString.format("Feedback for input %s", i));
            log.info(RefString.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
            log.info(RefString.format("Value Statistics: %s",
                inputPrototype[i].getScalarStatistics()));
            log.info(RefString.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
            log.info(RefString.format("Implemented Statistics: %s",
                implementedGradient.getScalarStatistics()));
            if (null != measuredGradient) {
              log.info(RefString.format("Measured: %s", measuredGradient.prettyPrint()));
              log.info(
                  RefString.format("Measured Statistics: %s", measuredGradient.getScalarStatistics()));
              Tensor temp_02_0032 = measuredGradient
                  .minus(implementedGradient.addRef());
              log.info(RefString.format("Feedback Error: %s", temp_02_0032.prettyPrint()));
              temp_02_0032.freeRef();
              Tensor temp_02_0033 = measuredGradient
                  .minus(implementedGradient.addRef());
              log.info(RefString.format("Error Statistics: %s", temp_02_0033.getScalarStatistics()));
              temp_02_0033.freeRef();
            }
            throw e;
          }
        }, IOPair, component, inputPrototype)).reduce(ToleranceStatistics::combine)));
    return statistics;
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    log.h1("Differential Validation");
    BatchDerivativeTester.IOPair temp_02_0019 = new IOPair(component.addRef(),
        inputPrototype[0].addRef(), BatchDerivativeTester.this.addRef());
    temp_02_0019.invoke();
    @Nonnull
    IOPair ioPair = temp_02_0019.addRef();

    temp_02_0019.freeRef();
    if (verbose) {
      log.run(RefUtil.wrapInterface(() -> {
        BatchDerivativeTester.log
            .info(RefString.format("Inputs: %s", RefUtil.get(RefArrays.stream(RefUtil.addRef(inputPrototype)).map(t -> {
              String temp_02_0007 = t.prettyPrint();
              t.freeRef();
              return temp_02_0007;
            }).reduce((a, b) -> a + ",\n" + b))));
        BatchDerivativeTester.log
            .info(RefString.format("Inputs Statistics: %s", RefUtil.get(RefArrays.stream(RefUtil.addRef(inputPrototype)).map(x -> {
              String temp_02_0008 = x.getScalarStatistics().toString();
              x.freeRef();
              return temp_02_0008;
            }).reduce((a, b) -> a + ",\n" + b))));
        Tensor temp_02_0034 = ioPair.getOutputPrototype();
        assert temp_02_0034 != null;
        BatchDerivativeTester.log.info(RefString.format("Output: %s", temp_02_0034.prettyPrint()));
        temp_02_0034.freeRef();
        Tensor temp_02_0035 = ioPair.getOutputPrototype();
        BatchDerivativeTester.log
            .info(RefString.format("Outputs Statistics: %s", temp_02_0035.getScalarStatistics()));
        temp_02_0035.freeRef();
      }, RefUtil.addRef(inputPrototype), ioPair.addRef()));
    }

    RefUtil.freeRef(inputPrototype);
    ToleranceStatistics _statistics = new ToleranceStatistics();

    if (isTestFeedback()) {
      log.h2("Feedback Validation");
      log.p(
          "We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.eval(RefUtil.wrapInterface(() -> {
        return testFeedback(component.addRef(), ioPair.addRef(),
            statistics);
      }, component.addRef(), ioPair.addRef()));
    }
    if (isTestLearning()) {
      log.h2("Learning Validation");
      log.p(
          "We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.eval(RefUtil.wrapInterface(() -> {
        return testLearning(component.addRef(), ioPair.addRef(),
            statistics);
      }, component.addRef(), ioPair.addRef()));
    }

    log.h2("Total Accuracy");
    log.p("The overall agreement accuracy between the implemented derivative and the finite difference estimations:");
    ToleranceStatistics statistics = _statistics;
    log.run(() -> {
      //log.info(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toStream(inputPrototype), outputPrototype));
      BatchDerivativeTester.log.info(RefString.format("Finite-Difference Derivative Accuracy:"));
      BatchDerivativeTester.log.info(RefString.format("absoluteTol: %s", statistics.absoluteTol));
      BatchDerivativeTester.log.info(RefString.format("relativeTol: %s", statistics.relativeTol));
    });

    log.h2("Frozen and Alive Status");
    log.run(RefUtil.wrapInterface(() -> {
      testFrozen(component.addRef(), ioPair.getInputPrototype());
      testUnFrozen(component.addRef(), ioPair.getInputPrototype());
    }, component, ioPair));

    return _statistics;
  }

  public void testFrozen(@Nonnull final Layer component, @Nonnull final Tensor[] inputPrototype) {
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_02_0036 = component.copy();
    temp_02_0036.freeze();
    @Nonnull final Layer frozen = temp_02_0036.addRef();
    temp_02_0036.freeRef();
    Result.Accumulator accumulator = new Result.Accumulator() {
      @Override
      public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
        buffer.freeRef();
        data.freeRef();
        reachedInputFeedback.set(true);
      }

      @Override
      public void _free() {
        super._free();
      }
    };
    @Nullable final Result eval = frozen
        .eval(new Result(new TensorArray(RefUtil.addRef(inputPrototype)), accumulator, true));
    frozen.freeRef();
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    assert eval != null;
    TensorList temp_02_0037 = eval.getData();
    TensorList tensorList = temp_02_0037.copy();
    temp_02_0037.freeRef();
    eval.accumulate(buffer.addRef(), tensorList == null ? null : tensorList.addRef());
    if (null != tensorList)
      tensorList.freeRef();
    eval.freeRef();
    RefList<double[]> temp_02_0038 = component.state();
    assert temp_02_0038 != null;
    final RefList<Delta<UUID>> deltas = temp_02_0038.stream()
        .map(RefUtil.wrapInterface((Function<? super double[], ? extends Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_02_0040 = buffer.stream().filter(x -> {
            boolean temp_02_0009 = x.target == doubles;
            x.freeRef();
            return temp_02_0009;
          }).findFirst();
          Delta<UUID> temp_02_0039 = temp_02_0040.orElse(null);
          RefUtil.freeRef(temp_02_0040);
          return temp_02_0039;
        }, buffer)).filter(x -> {
          boolean temp_02_0010 = x != null;
          if (null != x)
            x.freeRef();
          return temp_02_0010;
        }).collect(RefCollectors.toList());
    temp_02_0038.freeRef();
    RefList<double[]> temp_02_0041 = component.state();
    assert temp_02_0041 != null;
    try {
      if (!deltas.isEmpty() && !temp_02_0041.isEmpty()) {
        throw new AssertionError("Frozen component listed in evalInputDelta. Deltas: " + deltas);
      }
      final int inElements = RefArrays.stream(RefUtil.addRef(inputPrototype)).mapToInt(x -> {
        int temp_02_0012 = x.length();
        x.freeRef();
        return temp_02_0012;
      }).sum();
      if (!reachedInputFeedback.get() && 0 < inElements) {
        throw new RuntimeException("Frozen component did not pass input backwards");
      }
    } finally {
      deltas.freeRef();
      component.freeRef();
      RefUtil.freeRef(inputPrototype);
      temp_02_0041.freeRef();
    }
  }

  public void testUnFrozen(@Nonnull final Layer component, @Nullable final Tensor[] inputPrototype) {
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_02_0042 = component.copy();
    temp_02_0042.setFrozen(false);
    @Nonnull final Layer frozen = temp_02_0042.addRef();
    temp_02_0042.freeRef();
    component.freeRef();
    Result.Accumulator accumulator = new Result.Accumulator() {
      @Override
      public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
        buffer.freeRef();
        data.freeRef();
        reachedInputFeedback.set(true);
      }

      @Override
      public void _free() {
        super._free();
      }
    };
    @Nullable final Result eval = frozen.eval(new Result(new TensorArray(RefUtil.addRef(inputPrototype)), accumulator, true));
    if (null != inputPrototype)
      RefUtil.freeRef(inputPrototype);
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    assert eval != null;
    TensorList data = eval.getData();
    eval.accumulate(buffer.addRef(), data.addRef());
    data.freeRef();
    eval.freeRef();
    @Nullable final RefList<double[]> stateList = frozen.state();
    frozen.freeRef();
    assert stateList != null;
    final RefList<Delta<UUID>> deltas = stateList.stream()
        .map(RefUtil.wrapInterface((Function<? super double[], ? extends Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_02_0044 = buffer.stream().filter(x -> {
            boolean temp_02_0013 = x.target == doubles;
            x.freeRef();
            return temp_02_0013;
          }).findFirst();
          Delta<UUID> temp_02_0043 = temp_02_0044.orElse(null);
          RefUtil.freeRef(temp_02_0044);
          return temp_02_0043;
        }, buffer)).filter(x -> {
          boolean temp_02_0014 = x != null;
          if (null != x)
            x.freeRef();
          return temp_02_0014;
        }).collect(RefCollectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      stateList.freeRef();
      AssertionError temp_02_0015 = new AssertionError(
          "Nonfrozen component not listed in evalInputDelta. Deltas: " + deltas);
      deltas.freeRef();
      throw temp_02_0015;
    }
    deltas.freeRef();
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BatchDerivativeTester addRef() {
    return (BatchDerivativeTester) super.addRef();
  }

  @Nonnull
  private Tensor getFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                     @Nonnull final Tensor outputPrototype, @Nullable final Tensor... inputPrototype) {
    assert inputPrototype != null;
    final Tensor inputTensor = inputPrototype[inputIndex].addRef();
    final int inputDims = inputTensor.length();
    @Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor());
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          inputTensor.addRef();
          outputPrototype.addRef();
          inputKey.addRef();
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
          Tensor tensor = data.get(inputIndex);
          if (!RefArrays.equals(inputTensor.getDimensions(), tensor.getDimensions())) {
            buffer.freeRef();
            data.freeRef();
            tensor.freeRef();
            gradientBuffer.freeRef();
            throw new AssertionError();
          }
          for (int i = 0; i < inputDims; i++) {
            gradientBuffer.set(new int[]{i, j_}, tensor.get(i));
          }
          tensor.freeRef();
          Delta<UUID> delta = buffer.get(inputKey.getId(), new double[gradientBuffer.length()]);
          assert delta != null;
          delta.addInPlace(gradientBuffer);
          delta.freeRef();
          buffer.freeRef();
          data.freeRef();
        }

        @Override
        public void _free() {
          inputTensor.freeRef();
          outputPrototype.freeRef();
          inputKey.freeRef();
          super._free();
        }
      };
      @Nonnull final Result copyInput = new Result(new TensorArray(RefUtil.addRef(inputPrototype)), accumulator, true);
      @Nullable final Result eval = component.eval(copyInput);
      @Nonnull final DeltaSet<UUID> xxx = new DeltaSet<UUID>();
      assert eval != null;
      TensorList temp_02_0045 = eval.getData();
      @Nonnull
      TensorArray tensorArray = new TensorArray(temp_02_0045.stream().map(x -> {
        x.set(j_, 1);
        Tensor temp_02_0016 = x.addRef();
        x.freeRef();
        return temp_02_0016;
      }).toArray(Tensor[]::new));
      temp_02_0045.freeRef();
      eval.accumulate(xxx.addRef(), tensorArray);
      eval.freeRef();
      RefMap<UUID, Delta<UUID>> temp_02_0046 = xxx.getMap();
      final Delta<UUID> inputDelta = temp_02_0046.get(inputKey);
      temp_02_0046.freeRef();
      xxx.freeRef();
      if (null != inputDelta) {
        result.addInPlace(new Tensor(inputDelta.getDelta(), result.getDimensions()));
      }
      if (null != inputDelta)
        inputDelta.freeRef();
    }
    RefUtil.freeRef(inputPrototype);
    outputPrototype.freeRef();
    component.freeRef();
    inputTensor.freeRef();
    return result;
  }

  @Nonnull
  private Tensor getLearningGradient(@Nonnull final Layer component, final int layerNum,
                                     @Nonnull final Tensor outputPrototype, @Nullable final Tensor... inputPrototype) {
    component.setFrozen(false);
    RefList<double[]> temp_02_0047 = component.state();
    assert temp_02_0047 != null;
    final double[] stateArray = temp_02_0047.get(layerNum);
    temp_02_0047.freeRef();
    final int stateLen = stateArray.length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
      Tensor temp_02_0020 = new Tensor(outputPrototype.getDimensions());
      temp_02_0020.set(k -> k == j_ ? 1 : 0);
      @Nonnull final Tensor data = temp_02_0020.addRef();
      temp_02_0020.freeRef();
      @Nullable final Result eval = component
          .eval(ConstantResult.singleResultArray(new Tensor[][]{RefUtil.addRef(inputPrototype)}));
      assert eval != null;
      TensorList temp_02_0048 = eval.getData();
      RefUtil.freeRef(temp_02_0048.get(0));
      temp_02_0048.freeRef();
      @Nonnull
      TensorArray tensorArray = new TensorArray(data);
      eval.accumulate(buffer.addRef(), tensorArray);
      eval.freeRef();
      RefMap<UUID, Delta<UUID>> temp_02_0049 = buffer.getMap();
      RefCollection<Delta<UUID>> temp_02_0050 = temp_02_0049.values();
      Optional<Delta<UUID>> temp_02_0051 = temp_02_0050.stream().filter(x -> {
        boolean temp_02_0017 = x.target == stateArray;
        x.freeRef();
        return temp_02_0017;
      }).findFirst();
      final DoubleBuffer<UUID> deltaFlushBuffer = temp_02_0051.orElse(null);
      RefUtil.freeRef(temp_02_0051);
      temp_02_0050.freeRef();
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
      RefUtil.freeRef(inputPrototype);
    outputPrototype.freeRef();
    component.freeRef();
    return gradient;
  }

  @Nonnull
  private Tensor measureFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                         @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    @Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].length(), outputPrototype.length());
    Result temp_02_0052 = component
        .eval(ConstantResult.singleResultArray(new Tensor[][]{RefUtil.addRef(inputPrototype)}));
    assert temp_02_0052 != null;
    TensorList temp_02_0053 = temp_02_0052.getData();
    @Nullable final Tensor baseOutput = temp_02_0053.get(0);
    temp_02_0053.freeRef();
    temp_02_0052.freeRef();
    outputPrototype.set(baseOutput.addRef());
    outputPrototype.freeRef();
    for (int i = 0; i < inputPrototype[inputIndex].length(); i++) {
      @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      @Nonnull final Tensor[] copyInput = RefArrays.copyOf(RefUtil.addRef(inputPrototype), inputPrototype.length);
      RefUtil.set(copyInput, inputIndex, inputProbe);
      Result temp_02_0054 = component
          .eval(ConstantResult.singleResultArray(new Tensor[][]{copyInput}));
      assert temp_02_0054 != null;
      TensorList temp_02_0055 = temp_02_0054.getData();
      @Nullable final Tensor evalProbe = temp_02_0055.get(0);
      temp_02_0055.freeRef();
      temp_02_0054.freeRef();
      Tensor temp_02_0056 = evalProbe.minus(baseOutput.addRef());
      temp_02_0056.scaleInPlace(1. / probeSize);
      @Nonnull final Tensor delta = temp_02_0056.addRef();
      temp_02_0056.freeRef();
      evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.get(j));
      }
      delta.freeRef();
    }
    RefUtil.freeRef(inputPrototype);
    component.freeRef();
    baseOutput.freeRef();
    return measuredGradient;
  }

  @Nonnull
  private Tensor measureLearningGradient(@Nonnull final Layer component, final int layerNum,
                                         @Nonnull final Tensor outputPrototype, @Nullable final Tensor... inputPrototype) {
    RefList<double[]> temp_02_0057 = component.state();
    assert temp_02_0057 != null;
    double[] doubles = temp_02_0057.get(layerNum);
    final int stateLen = doubles.length;
    temp_02_0057.freeRef();
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    outputPrototype.freeRef();
    Result temp_02_0058 = component
        .eval(ConstantResult.singleResultArray(new Tensor[][]{RefUtil.addRef(inputPrototype)}));
    assert temp_02_0058 != null;
    TensorList temp_02_0059 = temp_02_0058.getData();
    @Nullable final Tensor baseOutput = temp_02_0059.get(0);
    temp_02_0059.freeRef();
    temp_02_0058.freeRef();
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      RefList<double[]> temp_02_0060 = copy.state();
      assert temp_02_0060 != null;
      double[] doubles1 = temp_02_0060.get(layerNum);
      doubles1[i] += probeSize;
      temp_02_0060.freeRef();
      Result temp_02_0061 = copy
          .eval(ConstantResult.singleResultArray(new Tensor[][]{RefUtil.addRef(inputPrototype)}));
      assert temp_02_0061 != null;
      TensorList temp_02_0062 = temp_02_0061.getData();
      @Nullable final Tensor evalProbe = temp_02_0062.get(0);
      temp_02_0062.freeRef();
      temp_02_0061.freeRef();
      copy.freeRef();
      Tensor delta = evalProbe.minus(baseOutput.addRef());
      delta.scaleInPlace(1. / probeSize);
      evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.get(j));
      }
      delta.freeRef();
    }
    if (null != inputPrototype)
      RefUtil.freeRef(inputPrototype);
    component.freeRef();
    baseOutput.freeRef();
    return gradient;
  }

  private static class IOPair extends ReferenceCountingBase {
    @Nullable
    private final Layer component;
    @Nonnull
    private final Tensor tensor;
    @Nullable
    private final BatchDerivativeTester parent;
    @Nullable
    private Tensor[] inputPrototype;
    @Nullable
    private Tensor outputPrototype;

    public IOPair(@Nullable Layer component, @Nullable Tensor tensor, @Nullable BatchDerivativeTester parent) {
      this.component = component;
      this.tensor = tensor;
      this.parent = parent;
    }

    @Nullable
    public Tensor[] getInputPrototype() {
      return RefUtil.addRef(inputPrototype);
    }

    @Nullable
    public Tensor getOutputPrototype() {
      return outputPrototype == null ? null : outputPrototype.addRef();
    }

    public void invoke() {
      assert parent != null;
      if (null != inputPrototype)
        RefUtil.freeRef(inputPrototype);
      inputPrototype = RefIntStream.range(0, parent.batches)
          .mapToObj(i -> tensor.copy())
          .toArray(Tensor[]::new);
      SimpleEval simpleEval = SimpleEval.run(component == null ? null : component.addRef(),
          inputPrototype[0].addRef());
      if (null != outputPrototype)
        outputPrototype.freeRef();
      outputPrototype = simpleEval.getOutput();
      simpleEval.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      if (null != outputPrototype)
        outputPrototype.freeRef();
      outputPrototype = null;
      if (null != inputPrototype)
        RefUtil.freeRef(inputPrototype);
      inputPrototype = null;
      if (null != parent)
        parent.freeRef();
      tensor.freeRef();
      if (null != component)
        component.freeRef();
    }

    @Nonnull
    public @Override
    @SuppressWarnings("unused")
    IOPair addRef() {
      return (IOPair) super.addRef();
    }
  }
}
