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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.data.ScalarStatistics;
import org.jetbrains.annotations.NotNull;
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

public class SingleDerivativeTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger log = LoggerFactory.getLogger(SingleDerivativeTester.class);

  public final double probeSize;
  private final double tolerance;
  private boolean testFeedback = true;
  private boolean testLearning = true;
  private boolean verbose = true;
  private boolean verify = true;

  public SingleDerivativeTester(final double tolerance, final double probeSize) {
    this.tolerance = tolerance;
    this.probeSize = probeSize;
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

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, @Nonnull final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    output.h1("Differential Validation");
    SimpleEval temp_00_0023 = SimpleEval.run(component.addRef(),
        RefUtil.addRefs(inputPrototype));
    final Tensor outputPrototype = temp_00_0023.getOutput();
    temp_00_0023.freeRef();
    try {
      if (verbose) {
        output.run(RefUtil.wrapInterface(() -> {
              log.info(RefString.format("Inputs: %s", prettyPrint(inputPrototype)));
              log.info(RefString.format("Inputs Statistics: %s", printStats(inputPrototype)));
              log.info(RefString.format("Output: %s", null == outputPrototype ? null : outputPrototype.prettyPrint()));
              assert outputPrototype != null;
              log.info(RefString.format("Outputs Statistics: %s", new ScalarStatistics().add(outputPrototype.getData())));
            },
            outputPrototype == null ? null : outputPrototype.addRef(),
            RefUtil.addRefs(inputPrototype)));
      }
      ToleranceStatistics _statistics = new ToleranceStatistics();
      if (isTestFeedback()) {
        output.h2("Feedback Validation");
        output.p(
            "We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
        final ToleranceStatistics statistics = _statistics;
        _statistics = output.eval(RefUtil.wrapInterface(() -> {
              return testFeedback(
                  statistics,
                  component.addRef(),
                  RefUtil.addRefs(inputPrototype),
                  outputPrototype == null ? null : outputPrototype.addRef());
            },
            outputPrototype == null ? null : outputPrototype.addRef(),
            RefUtil.addRefs(inputPrototype),
            component.addRef()));
      }
      if (isTestLearning()) {
        output.h2("Learning Validation");
        output.p(
            "We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
        final ToleranceStatistics statistics = _statistics;
        _statistics = output.eval(RefUtil.wrapInterface(() -> {
              return testLearning(
                  statistics,
                  component.addRef(),
                  RefUtil.addRefs(inputPrototype),
                  outputPrototype == null ? null : outputPrototype.addRef());
            },
            outputPrototype.addRef(),
            RefUtil.addRefs(inputPrototype),
            component.addRef()));
      }
      output.h2("Total Accuracy");
      output
          .p("The overall agreement accuracy between the implemented derivative and the finite difference estimations:");
      final ToleranceStatistics statistics = _statistics;
      output.run(() -> {
        //log.info(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toStream(inputPrototype), outputPrototype));
        log.info(RefString.format("Finite-Difference Derivative Accuracy:"));
        log.info(RefString.format("absoluteTol: %s", statistics.absoluteTol));
        log.info(RefString.format("relativeTol: %s", statistics.relativeTol));
      });

      output.h2("Frozen and Alive Status");
      output.run(RefUtil.wrapInterface(() -> {
        testFrozen(component.addRef(), RefUtil.addRefs(inputPrototype));
        testUnFrozen(component.addRef(), RefUtil.addRefs(inputPrototype));
      }, RefUtil.addRefs(inputPrototype), component.addRef()));
      return _statistics;
    } finally {
      outputPrototype.freeRef();
      component.freeRef();
      RefUtil.freeRefs(inputPrototype);
    }
  }

  @NotNull
  public String printStats(@RefIgnore Tensor[] array) {
    return Arrays.stream(array).map(Tensor::getData)
        .map(data -> new ScalarStatistics().add(data))
        .map(ScalarStatistics::toString)
        .reduce((a, b) -> a + ",\n" + b)
        .orElse("");
  }

  @NotNull
  public String prettyPrint(@RefIgnore Tensor[] array) {
    return Arrays.stream(array)
        .map(Tensor::prettyPrint)
        .reduce((a, b) -> a + ",\n" + b)
        .orElse("");
  }

  public ToleranceStatistics testLearning(@Nonnull ToleranceStatistics prev, @Nonnull Layer component,
                                          @Nullable Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    RefList<double[]> temp_00_0024 = component.state();
    assert temp_00_0024 != null;
    int size = temp_00_0024.size();
    temp_00_0024.freeRef();
    return RefIntStream.range(0, size)
        .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i -> {
              Tensor temp_00_0025 = measureLearningGradient(component.addRef(), i,
                  outputPrototype.addRef(), RefUtil.addRefs(inputPrototype));
              @Nullable final Tensor measuredGradient = !verify ? null : temp_00_0025.addRef();
              temp_00_0025.freeRef();
              @Nonnull final Tensor implementedGradient = getLearningGradient(component.addRef(), i,
                  outputPrototype.addRef(), RefUtil.addRefs(inputPrototype));
              assert measuredGradient != null;
              @Nonnull
              Tensor difference = measuredGradient.minus(implementedGradient.addRef());
              try {
                final ToleranceStatistics result = RefIntStream
                    .range(0, measuredGradient.length())
                    .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i1 -> {
                          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                              implementedGradient.getData()[i1]);
                        }, implementedGradient.addRef(),
                        measuredGradient.addRef()))
                    .reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
                if (!(result.absoluteTol.getMax() < tolerance)) {
                  measuredGradient.freeRef();
                  implementedGradient.freeRef();
                  difference.freeRef();
                  throw new AssertionError(result.toString());
                } else {
                  //log.info(String.format("Component: %s", component));
                  if (verbose) {

                    log.info(RefString.format("Learning Gradient for weight setByCoord %s", i));
                    RefList<double[]> temp_00_0026 = component.state();
                    assert temp_00_0026 != null;
                    log.info(RefString.format("Weights: %s", Tensor.prettyPrint(temp_00_0026.get(i))));
                    temp_00_0026.freeRef();
                    log.info(RefString.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
                    log.info(RefString.format("Implemented Statistics: %s",
                        new ScalarStatistics().add(implementedGradient.getData())));
                    log.info(RefString.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
                    log.info(RefString.format("Measured Statistics: %s",
                        new ScalarStatistics().add(measuredGradient.getData())));
                    log.info(RefString.format("Gradient Error: %s", difference.prettyPrint()));
                    log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                  }
                  measuredGradient.freeRef();
                  implementedGradient.freeRef();
                  difference.freeRef();
                  return result;
                }
              } catch (@Nonnull final Throwable e) {
                //log.info(String.format("Component: %s", component));
                log.info(RefString.format("Learning Gradient for weight setByCoord %s", i));
                log.info(RefString.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
                log.info(RefString.format("Implemented Statistics: %s",
                    new ScalarStatistics().add(implementedGradient.getData())));
                log.info(RefString.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
                log.info(
                    RefString.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                log.info(RefString.format("Gradient Error: %s", difference.prettyPrint()));
                log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                throw e;
              } finally {
              }
            }, outputPrototype, component, inputPrototype))
        .reduce((a, b) -> a.combine(b)).map(x -> x.combine(prev)).orElse(prev);
  }

  @Nonnull
  public ToleranceStatistics testFeedback(@Nonnull ToleranceStatistics statistics, @Nonnull Layer component,
                                          @Nonnull Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    Optional<ToleranceStatistics> optional = RefIntStream.range(0, inputPrototype.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i -> {
              assert verify;
              //@Nullable final Tensor measuredGradient = !verify ? null : temp_00_0027.addRef();
              final Tensor measuredGradient = measureFeedbackGradient(component.addRef(), i,
                  outputPrototype.addRef(), RefUtil.addRefs(inputPrototype));
              @Nonnull final Tensor implementedGradient = getFeedbackGradient(component.addRef(), i,
                  outputPrototype.addRef(), RefUtil.addRefs(inputPrototype));
              Tensor maskedGradient = implementedGradient.mapCoords(RefUtil.wrapInterface(
                  c -> {
                    return Double.isNaN(measuredGradient.get(c.getCoords())) ? Double.NaN : implementedGradient.get(c);
                  },
                  implementedGradient.addRef(),
                  measuredGradient.addRef()
              ));
              @Nonnull
              Tensor difference = measuredGradient.minus(maskedGradient.addRef());
              try {
                final ToleranceStatistics result = RefIntStream
                    .range(0, measuredGradient.length())
                    .mapToObj(RefUtil.wrapInterface((IntFunction<ToleranceStatistics>) i1 -> {
                          return new ToleranceStatistics().accumulate(
                              measuredGradient.getData()[i1],
                              maskedGradient.getData()[i1]);
                        },
                        maskedGradient.addRef(),
                        measuredGradient.addRef()
                    ))
                    .reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());

                if (!(result.absoluteTol.getMax() < tolerance)) {
                  measuredGradient.freeRef();
                  implementedGradient.freeRef();
                  maskedGradient.freeRef();
                  difference.freeRef();
                  throw new AssertionError(result.toString());
                }
                //log.info(String.format("Component: %s", component));
                if (verbose) {
                  log.info(RefString.format("Feedback for input %s", i));
                  log.info(RefString.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
                  log.info(
                      RefString.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
                  log.info(RefString.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
                  log.info(RefString.format("Implemented Statistics: %s",
                      new ScalarStatistics().add(implementedGradient.getData())));
                  log.info(RefString.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
                  log.info(RefString.format("Measured Statistics: %s",
                      new ScalarStatistics().add(measuredGradient.getData())));
                  log.info(RefString.format("Feedback Error: %s", difference.prettyPrint()));
                  log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                }

                measuredGradient.freeRef();
                implementedGradient.freeRef();
                maskedGradient.freeRef();
                difference.freeRef();
                return result;
              } catch (@Nonnull final Throwable e) {
                //log.info(String.format("Component: %s", component));
                log.info(RefString.format("Feedback for input %s", i));
                log.info(RefString.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
                log.info(RefString.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
                if (!implementedGradient.isFinalized()) {
                  log.info(RefString.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
                  log.info(RefString.format("Implemented Statistics: %s",
                      new ScalarStatistics().add(implementedGradient.getData())));
                }
                if (!measuredGradient.isFinalized()) {
                  log.info(RefString.format("Measured: %s", measuredGradient.prettyPrint()));
                  log.info(RefString.format(
                      "Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                }
                if (!difference.isFinalized()) {
                  log.info(RefString.format("Feedback Error: %s", difference.prettyPrint()));
                  log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                }
                throw e;
              }
            },
            component,
            inputPrototype,
            outputPrototype
        )).reduce((a, b) -> a.combine(b));
    if (!optional.isPresent())
      return statistics;
    return statistics.combine(RefUtil.orElse(optional, null));
  }

  public void testFrozen(@Nonnull final Layer component, @Nonnull Tensor[] inputPrototype) {
    final int inElements = RefArrays.stream(RefUtil.addRefs(inputPrototype)).mapToInt(x -> {
      int temp_00_0005 = x.length();
      x.freeRef();
      return temp_00_0005;
    }).sum();
    inputPrototype = RefArrays.stream(inputPrototype).map(tensor -> {
      Tensor temp_00_0006 = tensor.copy();
      tensor.freeRef();
      return temp_00_0006;
    }).toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_00_0028 = component.copy();
    temp_00_0028.freeze();
    @Nonnull final Layer frozen = temp_00_0028.addRef();
    temp_00_0028.freeRef();
    RefList<TensorArray> inputCopies = RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(data -> {
      TensorArray temp_00_0007 = new TensorArray(data == null ? null : data.addRef());
      if (null != data)
        data.freeRef();
      return temp_00_0007;
    }).collect(RefCollectors.toList());
    ReferenceCounting.freeRefs(inputPrototype);
    Result[] input = inputCopies.stream().map((tensorArray) -> {
      try {
        Result.Accumulator accumulator = new Result.Accumulator() {
          @Override
          public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
            reachedInputFeedback.set(true);
            buffer.freeRef();
            data.freeRef();
          }

          @Override
          public void _free() {
            super._free();
          }
        };
        return new Result(tensorArray, accumulator) {
          @Override
          public boolean isAlive() {
            return true;
          }

          @Override
          public void _free() {
            super._free();
          }
        };
      } finally {
        if (null != tensorArray)
          tensorArray.freeRef();
      }
    }).toArray(i -> new Result[i]);
    inputCopies.freeRef();
    @Nullable final Result eval = frozen.eval(RefUtil.addRefs(input));
    ReferenceCounting.freeRefs(input);
    frozen.freeRef();
    assert eval != null;
    TensorList evalData = eval.getData();
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList tensorList = evalData.copy();
    eval.accumulate(buffer.addRef(), tensorList == null ? null : tensorList.addRef());
    evalData.freeRef();
    if (null != tensorList)
      tensorList.freeRef();
    eval.freeRef();
    RefList<double[]> temp_00_0029 = component.state();
    assert temp_00_0029 != null;
    final RefList<Delta<UUID>> deltas = temp_00_0029.stream()
        .map(RefUtil.wrapInterface((Function<double[], Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_00_0031 = buffer.stream().filter(x -> {
            boolean temp_00_0009 = x.target == doubles;
            x.freeRef();
            return temp_00_0009;
          }).findFirst();
          Delta<UUID> temp_00_0030 = temp_00_0031.orElse(null);
          RefUtil.freeRef(temp_00_0031);
          return temp_00_0030;
        }, buffer)).filter(x -> {
          boolean temp_00_0010 = x != null;
          if (null != x)
            x.freeRef();
          return temp_00_0010;
        }).collect(RefCollectors.toList());
    temp_00_0029.freeRef();
    RefList<double[]> temp_00_0032 = component.state();
    assert temp_00_0032 != null;
    if (!deltas.isEmpty() && !temp_00_0032.isEmpty()) {
      temp_00_0032.freeRef();
      AssertionError temp_00_0011 = new AssertionError("Frozen component listed in evalInputDelta. Deltas: " + deltas);
      deltas.freeRef();
      component.freeRef();
      throw temp_00_0011;
    }
    temp_00_0032.freeRef();
    component.freeRef();
    deltas.freeRef();
    if (!reachedInputFeedback.get() && 0 < inElements) {
      throw new RuntimeException("Frozen component did not pass input backwards");
    }
  }

  public void testUnFrozen(@Nonnull final Layer component, Tensor[] inputPrototype) {
    inputPrototype = RefArrays.stream(inputPrototype).map(tensor -> {
      Tensor temp_00_0012 = tensor.copy();
      tensor.freeRef();
      return temp_00_0012;
    }).toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_00_0033 = component.copy();
    temp_00_0033.setFrozen(false);
    @Nonnull final Layer frozen = temp_00_0033.addRef();
    temp_00_0033.freeRef();
    component.freeRef();
    RefList<TensorArray> inputCopies = RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(data -> {
      TensorArray temp_00_0013 = new TensorArray(data == null ? null : data.addRef());
      if (null != data)
        data.freeRef();
      return temp_00_0013;
    }).collect(RefCollectors.toList());
    Result[] inputs = inputCopies.stream().map(tensor -> {
      try {
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
        return new Result(tensor, accumulator) {
          @Override
          public boolean isAlive() {
            return true;
          }

          @Override
          public void _free() {
            super._free();
          }
        };
      } finally {
        if (null != tensor)
          tensor.freeRef();
      }
    }).toArray(i -> new Result[i]);
    inputCopies.freeRef();
    @Nullable final Result eval = frozen.eval(RefUtil.addRefs(inputs));
    ReferenceCounting.freeRefs(inputs);
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    assert eval != null;
    TensorList tensorList = eval.getData();
    eval.accumulate(buffer.addRef(), tensorList.addRef());
    tensorList.freeRef();
    eval.freeRef();
    @Nullable final RefList<double[]> stateList = frozen.state();
    frozen.freeRef();
    assert stateList != null;
    final RefList<Delta<UUID>> deltas = stateList.stream()
        .map(RefUtil.wrapInterface((Function<double[], Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_00_0035 = buffer.stream().filter(x -> {
            boolean temp_00_0015 = x.target == doubles;
            x.freeRef();
            return temp_00_0015;
          }).findFirst();
          Delta<UUID> temp_00_0034 = temp_00_0035.orElse(null);
          RefUtil.freeRef(temp_00_0035);
          return temp_00_0034;
        }, buffer)).filter(x -> {
          boolean temp_00_0016 = x != null;
          if (null != x)
            x.freeRef();
          return temp_00_0016;
        }).collect(RefCollectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      stateList.freeRef();
      AssertionError temp_00_0017 = new AssertionError(
          "Nonfrozen component not listed in evalInputDelta. Deltas: " + deltas);
      deltas.freeRef();
      ReferenceCounting.freeRefs(inputPrototype);
      throw temp_00_0017;
    }
    deltas.freeRef();
    stateList.freeRef();
    if (!reachedInputFeedback.get() && inputPrototype.length != 0) {
      ReferenceCounting.freeRefs(inputPrototype);
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
    ReferenceCounting.freeRefs(inputPrototype);
  }

  @Nonnull
  @Override
  public String toString() {
    return "SingleDerivativeTester{" + "probeSize=" + probeSize + ", tolerance=" + tolerance + ", testFeedback="
        + testFeedback + ", testLearning=" + testLearning + ", verbose=" + verbose + ", verify=" + verify + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SingleDerivativeTester addRef() {
    return (SingleDerivativeTester) super.addRef();
  }

  protected void measureFeedback(@Nonnull Layer component, int inputIndex, @Nullable Tensor baseOutput,
                                 @Nonnull Tensor[] inputPrototype, @Nonnull Tensor measuredGradient, int probeIndex) {
    @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
    inputProbe.add(probeIndex, probeSize * 1);
    @Nonnull final Tensor[] copyInput = RefArrays.copyOf(inputPrototype, inputPrototype.length);
    RefUtil.set(copyInput, inputIndex, inputProbe);
    try {
      Result temp_00_0036 = component.eval(ConstantResult.batchResultArray(new Tensor[][]{copyInput}));
      assert temp_00_0036 != null;
      TensorList temp_00_0037 = temp_00_0036.getData();
      @Nullable final Tensor evalProbe = temp_00_0037.get(0);
      temp_00_0037.freeRef();
      temp_00_0036.freeRef();
      Tensor delta = evalProbe.minus(baseOutput == null ? null : baseOutput.addRef());
      delta.scaleInPlace(1. / probeSize);
      evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        measuredGradient.set(new int[]{probeIndex, j}, delta.getData()[j]);
      }
      delta.freeRef();
    } finally {
      measuredGradient.freeRef();
      if (null != baseOutput)
        baseOutput.freeRef();
      component.freeRef();
    }
  }

  @Nonnull
  private Tensor getFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                     @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex].addRef();
    final int inputDims = inputTensor.length();
    @Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor(1));
      final Result[] copyInput = RefArrays.stream(RefUtil.addRefs(inputPrototype)).map(x -> {
        try {
          TensorArray data = new TensorArray(x == null ? null : x.addRef());
          Result.Accumulator accumulator = new Result.Accumulator() {
            @Override
            public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
              buffer.freeRef();
              data.freeRef();
            }

            @Override
            public void _free() {
              super._free();
            }
          };
          return new Result(data, accumulator);
        } finally {
          if (null != x)
            x.freeRef();
        }
      }).toArray(i -> new Result[i]);
      double[] target = new double[inputDims * outputPrototype.length()];
      Result.Accumulator accumulator = new Result.Accumulator() {

        {
          outputPrototype.addRef();
          inputKey.addRef();
          inputTensor.addRef();
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          try {
            if (1 != data.length())
              throw new AssertionError();
            if (data.length() != 1)
              throw new AssertionError();
            if (!RefArrays.equals(inputTensor.getDimensions(), data.getDimensions())) {
              throw new AssertionError();
            }
            @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
            RefIntStream.range(0, data.length()).forEach(dataIndex -> {
              for (int i = 0; i < inputDims; i++) {
                @Nullable
                Tensor tensor = data.get(dataIndex);
                gradientBuffer.set(new int[]{i, j_}, tensor.getData()[i]);
                tensor.freeRef();
              }
            });
            Delta<UUID> delta = buffer.get(inputKey.getId(), target);
            assert delta != null;
            delta.addInPlace(gradientBuffer.getData());
            gradientBuffer.freeRef();
            delta.freeRef();
          } finally {
            data.freeRef();
            buffer.freeRef();
          }
        }

        @Override
        public void _free() {
          outputPrototype.freeRef();
          inputKey.freeRef();
          inputTensor.freeRef();
          super._free();
        }
      };
      Result result1 = new Result(new TensorArray(inputTensor.addRef()), accumulator) {

        {
          inputKey.addRef();
        }

        @Override
        public boolean isAlive() {
          return true;
        }

        public @SuppressWarnings("unused")
        void _free() {
          inputKey.freeRef();
          super._free();
        }
      };
      RefUtil.set(copyInput, inputIndex, result1);
      @Nullable final Result eval = eval(component.addRef(), copyInput);
      @Nonnull final DeltaSet<UUID> deltaSet = new DeltaSet<UUID>();
      Tensor temp_00_0021 = new Tensor(outputPrototype.getDimensions());
      temp_00_0021.set(j, 1);
      @Nonnull
      TensorArray tensorArray = new TensorArray(temp_00_0021);
      try {
        assert eval != null;
        eval.accumulate(deltaSet.addRef(), tensorArray.addRef());
        RefMap<UUID, Delta<UUID>> map = deltaSet.getMap();
        final Delta<UUID> inputDelta = map.get(inputKey.getId());
        map.freeRef();
        if (null != inputDelta) {
          @Nonnull
          Tensor tensor = new Tensor(inputDelta.getDelta(), result.getDimensions());
          result.addInPlace(tensor);
        }
        if (null != inputDelta)
          inputDelta.freeRef();
      } finally {
        assert eval != null;
        RefUtil.freeRef(eval.getData());
        tensorArray.freeRef();
        deltaSet.freeRef();
        eval.freeRef();
        inputKey.freeRef();
      }
    }
    ReferenceCounting.freeRefs(inputPrototype);
    outputPrototype.freeRef();
    component.freeRef();
    inputTensor.freeRef();
    return result;
  }

  private Result eval(@Nonnull Layer component, Result[] copyInput) {
    try {
      return component.eval(RefUtil.addRefs(copyInput));
    } finally {
      component.freeRef();
      RefUtil.freeRefs(copyInput);
    }
  }

  @Nonnull
  private Tensor getLearningGradient(@Nonnull final Layer component, final int layerNum,
                                     @Nonnull final Tensor outputPrototype, @Nullable final Tensor... inputPrototype) {
    component.setFrozen(false);
    RefList<double[]> temp_00_0039 = component.state();
    assert temp_00_0039 != null;
    final double[] stateArray = temp_00_0039.get(layerNum);
    temp_00_0039.freeRef();
    final int stateLen = stateArray.length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
      Result[] array = ConstantResult.batchResultArray(new Tensor[][]{RefUtil.addRefs(inputPrototype)});
      @Nullable final Result eval = component.eval(array);
      Tensor temp_00_0022 = new Tensor(outputPrototype.getDimensions());
      temp_00_0022.set((k) -> k == j_ ? 1 : 0);
      @Nonnull
      TensorArray tensorArray = new TensorArray(temp_00_0022.addRef());
      temp_00_0022.freeRef();
      assert eval != null;
      eval.accumulate(buffer.addRef(), tensorArray);
      RefUtil.freeRef(eval.getData());
      eval.freeRef();
      RefMap<UUID, Delta<UUID>> temp_00_0040 = buffer.getMap();
      RefCollection<Delta<UUID>> temp_00_0041 = temp_00_0040.values();
      Optional<Delta<UUID>> temp_00_0042 = temp_00_0041.stream().filter(x -> {
        boolean temp_00_0019 = x.target == stateArray;
        x.freeRef();
        return temp_00_0019;
      }).findFirst();
      final DoubleBuffer<UUID> deltaFlushBuffer = temp_00_0042.orElse(null);
      RefUtil.freeRef(temp_00_0042);
      temp_00_0041.freeRef();
      temp_00_0040.freeRef();
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
    int length = inputPrototype[inputIndex].length();
    @Nonnull final Tensor measuredGradient = new Tensor(length, outputPrototype.length());
    Result[] input0 = ConstantResult.batchResultArray(new Tensor[][]{RefUtil.addRefs(inputPrototype)});
    Result temp_00_0043 = component.eval(input0);
    assert temp_00_0043 != null;
    TensorList temp_00_0044 = temp_00_0043.getData();
    temp_00_0043.freeRef();
    @Nullable final Tensor baseOutput = temp_00_0044.get(0);
    temp_00_0044.freeRef();
    outputPrototype.set(baseOutput.addRef());
    outputPrototype.freeRef();
    for (int probeIndex = 0; probeIndex < length; probeIndex++) {
      measureFeedback(component.addRef(), inputIndex,
          baseOutput.addRef(), RefUtil.addRefs(inputPrototype),
          measuredGradient.addRef(), probeIndex);
    }
    ReferenceCounting.freeRefs(inputPrototype);
    component.freeRef();
    baseOutput.freeRef();
    return measuredGradient;
  }

  @Nonnull
  private Tensor measureLearningGradient(@Nonnull final Layer component, final int layerNum,
                                         @Nonnull final Tensor outputPrototype, @Nullable final Tensor... inputPrototype) {
    RefList<double[]> temp_00_0045 = component.state();
    assert temp_00_0045 != null;
    final int stateLen = temp_00_0045.get(layerNum).length;
    temp_00_0045.freeRef();
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());

    outputPrototype.freeRef();
    Result[] input2 = ConstantResult.batchResultArray(new Tensor[][]{RefUtil.addRefs(inputPrototype)});
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    Result temp_00_0046 = component.eval(RefUtil.addRefs(input2));
    assert temp_00_0046 != null;
    TensorList temp_00_0047 = temp_00_0046.getData();
    @Nullable final Tensor baseOutput = temp_00_0047.get(0);

    temp_00_0047.freeRef();
    temp_00_0046.freeRef();
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      RefList<double[]> temp_00_0048 = copy.state();
      assert temp_00_0048 != null;
      temp_00_0048.get(layerNum)[i] += probeSize;
      temp_00_0048.freeRef();
      Result temp_00_0049 = copy.eval(RefUtil.addRefs(input2));
      assert temp_00_0049 != null;
      TensorList temp_00_0050 = temp_00_0049.getData();
      @Nullable final Tensor evalProbe = temp_00_0050.get(0);
      temp_00_0050.freeRef();
      temp_00_0049.freeRef();
      copy.freeRef();
      Tensor temp_00_0051 = evalProbe.minus(baseOutput.addRef());
      temp_00_0051.scaleInPlace(1. / probeSize);
      @Nonnull final Tensor delta = temp_00_0051.addRef();
      temp_00_0051.freeRef();
      evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    component.freeRef();
    baseOutput.freeRef();
    ReferenceCounting.freeRefs(input2);
    return gradient;
  }
}
