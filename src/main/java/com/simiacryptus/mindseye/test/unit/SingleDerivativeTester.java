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
import com.simiacryptus.util.data.ScalarStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

public @com.simiacryptus.ref.lang.RefAware
class SingleDerivativeTester extends ComponentTestBase<ToleranceStatistics> {
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

  @Nonnull
  public SingleDerivativeTester setTestFeedback(final boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this;
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  @Nonnull
  public SingleDerivativeTester setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  @Nonnull
  public SingleDerivativeTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public boolean isVerify() {
    return verify;
  }

  @Nonnull
  public SingleDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this;
  }

  public static @SuppressWarnings("unused")
  SingleDerivativeTester[] addRefs(SingleDerivativeTester[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SingleDerivativeTester::addRef)
        .toArray((x) -> new SingleDerivativeTester[x]);
  }

  public static @SuppressWarnings("unused")
  SingleDerivativeTester[][] addRefs(SingleDerivativeTester[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SingleDerivativeTester::addRefs)
        .toArray((x) -> new SingleDerivativeTester[x][]);
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, @Nonnull final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    output.h1("Differential Validation");
    ToleranceStatistics _statistics = new ToleranceStatistics();
    final Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutput();
    {
      if (verbose) {
        output.run(() -> {
          log.info(String.format("Inputs: %s", com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype)
              .map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).orElse("")));
          log.info(String.format("Inputs Statistics: %s",
              com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype)
                  .map(x -> new ScalarStatistics().add(x.getData()).toString()).reduce((a, b) -> a + ",\n" + b)
                  .orElse("")));
          log.info(String.format("Output: %s", null == outputPrototype ? null : outputPrototype.prettyPrint()));
          log.info(String.format("Outputs Statistics: %s", new ScalarStatistics().add(outputPrototype.getData())));
        });
      }
      if (isTestFeedback()) {
        output.h2("Feedback Validation");
        output.p(
            "We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
        final ToleranceStatistics statistics = _statistics;
        _statistics = output.eval(() -> {
          return testFeedback(statistics, component, inputPrototype, outputPrototype);
        });
      }
      if (isTestLearning()) {
        output.h2("Learning Validation");
        output.p(
            "We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
        final ToleranceStatistics statistics = _statistics;
        _statistics = output.eval(() -> {
          return testLearning(statistics, component, inputPrototype, outputPrototype);
        });
      }
    }
    output.h2("Total Accuracy");
    output
        .p("The overall agreement accuracy between the implemented derivative and the finite difference estimations:");
    final ToleranceStatistics statistics = _statistics;
    output.run(() -> {
      //log.info(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toStream(inputPrototype), outputPrototype));
      log.info(String.format("Finite-Difference Derivative Accuracy:"));
      log.info(String.format("absoluteTol: %s", statistics.absoluteTol));
      log.info(String.format("relativeTol: %s", statistics.relativeTol));
    });

    output.h2("Frozen and Alive Status");
    output.run(() -> {
      testFrozen(component, inputPrototype);
      testUnFrozen(component, inputPrototype);
    });

    return _statistics;
  }

  public ToleranceStatistics testLearning(@Nonnull ToleranceStatistics prev, @Nonnull Layer component,
                                          Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    return com.simiacryptus.ref.wrappers.RefIntStream.range(0, component.state().size()).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null
          : measureLearningGradient(component, i, outputPrototype, inputPrototype);
      @Nonnull final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
      @Nonnull
      Tensor difference = measuredGradient.minus(implementedGradient);
      try {
        final ToleranceStatistics result = com.simiacryptus.ref.wrappers.RefIntStream
            .range(0, null == measuredGradient ? 0 : measuredGradient.length()).mapToObj(i1 -> {
              return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                  implementedGradient.getData()[i1]);
            }).reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
        if (!(result.absoluteTol.getMax() < tolerance)) {
          throw new AssertionError(result.toString());
        } else {
          //log.info(String.format("Component: %s", component));
          if (verbose) {

            log.info(String.format("Learning Gradient for weight setByCoord %s", i));
            log.info(String.format("Weights: %s", Tensor.prettyPrint(component.state().get(i))));
            log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
            log.info(
                String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
              log.info(
                  String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              log.info(String.format("Gradient Error: %s", difference.prettyPrint()));
              log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
            }
          }
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
          log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
          log.info(String.format("Gradient Error: %s", difference.prettyPrint()));
          log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
        }
        throw e;
      } finally {
      }

    }).reduce((a, b) -> a.combine(b)).map(x -> x.combine(prev)).orElse(prev);
  }

  @Nonnull
  public ToleranceStatistics testFeedback(@Nonnull ToleranceStatistics statistics, @Nonnull Layer component,
                                          @Nonnull Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    Optional<ToleranceStatistics> optional = com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputPrototype.length)
        .mapToObj(i -> {
          @Nullable final Tensor measuredGradient = !verify ? null
              : measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
          @Nonnull final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
          Tensor maskedGradient = implementedGradient.mapCoords(
              c -> Double.isNaN(measuredGradient.get(c.getCoords())) ? Double.NaN : implementedGradient.get(c));
          @Nonnull
          Tensor difference = measuredGradient.minus(maskedGradient);
          try {
            final ToleranceStatistics result = com.simiacryptus.ref.wrappers.RefIntStream
                .range(0, null == measuredGradient ? 0 : measuredGradient.length()).mapToObj(i1 -> {
                  return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                      maskedGradient.getData()[i1]);
                }).reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());

            if (!(result.absoluteTol.getMax() < tolerance))
              throw new AssertionError(result.toString());
            //log.info(String.format("Component: %s", component));
            if (verbose) {
              log.info(String.format("Feedback for input %s", i));
              log.info(String.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
              log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
              log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
              log.info(String.format("Implemented Statistics: %s",
                  new ScalarStatistics().add(implementedGradient.getData())));
              if (null != measuredGradient) {
                log.info(String.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
                log.info(
                    String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                log.info(String.format("Feedback Error: %s", difference.prettyPrint()));
                log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
              }
            }

            return result;
          } catch (@Nonnull final Throwable e) {
            //log.info(String.format("Component: %s", component));
            log.info(String.format("Feedback for input %s", i));
            log.info(String.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
            log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
            log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
            log.info(
                String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured: %s", measuredGradient.prettyPrint()));
              log.info(
                  String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              log.info(String.format("Feedback Error: %s", difference.prettyPrint()));
              log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
            }
            throw e;
          } finally {
          }
        }).reduce((a, b) -> a.combine(b));
    if (!optional.isPresent())
      return statistics;
    return statistics.combine(optional.orElse(null));
  }

  public void testFrozen(@Nonnull final Layer component, @Nonnull Tensor[] inputPrototype) {
    final int inElements = com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype).mapToInt(x -> x.length())
        .sum();
    inputPrototype = com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype).map(tensor -> tensor.copy())
        .toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().freeze();
    com.simiacryptus.ref.wrappers.RefList<TensorArray> inputCopies = com.simiacryptus.ref.wrappers.RefArrays
        .stream(inputPrototype).map(data -> new TensorArray(data))
        .collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    Result[] input = inputCopies.stream().map((tensorArray) -> new Result(tensorArray,
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          reachedInputFeedback.set(true);
        }) {

      @Override
      public boolean isAlive() {
        return true;
      }

      public @SuppressWarnings("unused")
      void _free() {
      }

    }).toArray(i -> new Result[i]);
    @Nullable final Result eval;
    eval = frozen.eval(input);
    @Nonnull final DeltaSet<UUID> buffer;
    TensorList tensorList;
    TensorList evalData = eval.getData();
    {
      buffer = new DeltaSet<UUID>();
      tensorList = evalData.copy();
      eval.accumulate(buffer, tensorList);
    }
    final com.simiacryptus.ref.wrappers.RefList<Delta<UUID>> deltas = component.state().stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    if (!deltas.isEmpty() && !component.state().isEmpty()) {
      throw new AssertionError("Frozen component listed in evalInputDelta. Deltas: " + deltas);
    }
    if (!reachedInputFeedback.get() && 0 < inElements) {
      throw new RuntimeException("Frozen component did not pass input backwards");
    }
  }

  public void testUnFrozen(@Nonnull final Layer component, Tensor[] inputPrototype) {
    inputPrototype = com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype).map(tensor -> tensor.copy())
        .toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().setFrozen(false);
    com.simiacryptus.ref.wrappers.RefList<TensorArray> inputCopies = com.simiacryptus.ref.wrappers.RefArrays
        .stream(inputPrototype).map(data -> new TensorArray(data))
        .collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    Result[] inputs = inputCopies.stream()
        .map(tensor -> new Result(tensor, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          reachedInputFeedback.set(true);
        }) {
          @Override
          public boolean isAlive() {
            return true;
          }

          public @SuppressWarnings("unused")
          void _free() {
          }
        }).toArray(i -> new Result[i]);
    @Nullable final Result eval;
    eval = frozen.eval(inputs);
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList tensorList = eval.getData();
    eval.accumulate(buffer, tensorList);
    @Nullable final com.simiacryptus.ref.wrappers.RefList<double[]> stateList = frozen.state();
    final com.simiacryptus.ref.wrappers.RefList<Delta<UUID>> deltas = stateList.stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      throw new AssertionError("Nonfrozen component not listed in evalInputDelta. Deltas: " + deltas);
    }
    if (!reachedInputFeedback.get() && inputPrototype.length != 0) {
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
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

  public @Override
  @SuppressWarnings("unused")
  SingleDerivativeTester addRef() {
    return (SingleDerivativeTester) super.addRef();
  }

  protected void measureFeedback(@Nonnull Layer component, int inputIndex, Tensor baseOutput,
                                 @Nonnull Tensor[] inputPrototype, Tensor measuredGradient, int probeIndex) {
    @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
    inputProbe.add(probeIndex, probeSize * 1);
    @Nonnull final Tensor[] copyInput = com.simiacryptus.ref.wrappers.RefArrays.copyOf(inputPrototype, inputPrototype.length);
    copyInput[inputIndex] = inputProbe;
    Result[] input1 = ConstantResult.batchResultArray(new Tensor[][]{copyInput});
    try {
      @Nullable final Tensor evalProbe = component.eval(input1).getData().get(0);
      @Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.length(); j++) {
        measuredGradient.set(new int[]{probeIndex, j}, delta.getData()[j]);
      }
    } finally {
      for (@Nonnull
          Result result : input1) {
        result.getData();
      }

    }
  }

  @Nonnull
  private Tensor getFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                     @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex];
    final int inputDims = inputTensor.length();
    @Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor(1));
      inputKey.getKey();
      final Result[] copyInput = com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype)
          .map(x -> new Result(new TensorArray(x),
              (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
              }) {

            @Override
            public boolean isAlive() {
              return false;
            }

            public @SuppressWarnings("unused")
            void _free() {
            }

          }).toArray(i -> new Result[i]);
      copyInput[inputIndex].getData();
      double[] target = new double[inputDims * outputPrototype.length()];
      copyInput[inputIndex] = new Result(new TensorArray(inputTensor),
          (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
            {
              if (1 != data.length())
                throw new AssertionError();
              if (data.length() != 1)
                throw new AssertionError();
              @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
              if (!com.simiacryptus.ref.wrappers.RefArrays.equals(inputTensor.getDimensions(), data.getDimensions())) {
                throw new AssertionError();
              }
              com.simiacryptus.ref.wrappers.RefIntStream.range(0, data.length()).forEach(dataIndex -> {
                for (int i = 0; i < inputDims; i++) {
                  @Nullable
                  Tensor tensor = data.get(dataIndex);
                  gradientBuffer.set(new int[]{i, j_}, tensor.getData()[i]);
                }
              });
              buffer.get(inputKey.getId(), target).addInPlace(gradientBuffer.getData());
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
      @Nullable final Result eval;
      try {
        eval = component.eval(copyInput);
      } finally {
        for (@Nonnull
            Result nnResult : copyInput) {
          nnResult.getData();
        }
      }
      @Nonnull final DeltaSet<UUID> deltaSet = new DeltaSet<UUID>();
      @Nonnull
      TensorArray tensorArray = new TensorArray(new Tensor(outputPrototype.getDimensions()).set(j, 1));
      try {
        eval.accumulate(deltaSet, tensorArray);
        com.simiacryptus.ref.wrappers.RefMap<UUID, Delta<UUID>> map = deltaSet.getMap();
        final Delta<UUID> inputDelta = map.get(inputKey.getId());
        if (null != inputDelta) {
          @Nonnull
          Tensor tensor = new Tensor(inputDelta.getDelta(), result.getDimensions());
          result.addInPlace(tensor);
        }
      } finally {
        eval.getData();
      }
    }
    return result;
  }

  @Nonnull
  private Tensor getLearningGradient(@Nonnull final Layer component, final int layerNum,
                                     @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    component.setFrozen(false);
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
      Result[] array = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
      @Nullable final Result eval = component.eval(array);
      for (@Nonnull
          Result result : array) {
        result.getData();
      }
      @Nonnull
      TensorArray tensorArray = new TensorArray(
          new Tensor(outputPrototype.getDimensions()).set((k) -> k == j_ ? 1 : 0));
      eval.accumulate(buffer, tensorArray);
      eval.getData();
      final DoubleBuffer<UUID> deltaFlushBuffer = buffer.getMap().values().stream().filter(x -> x.target == stateArray)
          .findFirst().orElse(null);
      if (null != deltaFlushBuffer) {
        for (int i = 0; i < stateLen; i++) {
          gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
        }
      }
    }
    return gradient;
  }

  @Nonnull
  private Tensor measureFeedbackGradient(@Nonnull final Layer component, final int inputIndex,
                                         @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    @Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].length(), outputPrototype.length());
    Result[] input0 = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
    @Nullable final Tensor baseOutput = component.eval(input0).getData().get(0);
    for (@Nonnull
        Result result : input0) {
      result.getData();
    }
    outputPrototype.set(baseOutput);
    for (int probeIndex = 0; probeIndex < inputPrototype[inputIndex].length(); probeIndex++) {
      measureFeedback(component, inputIndex, baseOutput, inputPrototype, measuredGradient, probeIndex);
    }
    return measuredGradient;
  }

  @Nonnull
  private Tensor measureLearningGradient(@Nonnull final Layer component, final int layerNum,
                                         @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());

    Result[] input2 = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
    @Nullable final Tensor baseOutput = component.eval(input2).getData().get(0);

    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      copy.state().get(layerNum)[i] += probeSize;
      @Nullable final Tensor evalProbe = copy.eval(input2).getData().get(0);
      @Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    for (@Nonnull
        Result result : input2) {
      result.getData();
    }
    return gradient;
  }
}
