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
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.util.data.ScalarStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

public @com.simiacryptus.ref.lang.RefAware
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
    return this;
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  @Nonnull
  public BatchDerivativeTester setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  @Nonnull
  public BatchDerivativeTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public boolean isVerify() {
    return verify;
  }

  @Nonnull
  public BatchDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this;
  }

  public static @SuppressWarnings("unused")
  BatchDerivativeTester[] addRefs(BatchDerivativeTester[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BatchDerivativeTester::addRef)
        .toArray((x) -> new BatchDerivativeTester[x]);
  }

  public static @SuppressWarnings("unused")
  BatchDerivativeTester[][] addRefs(BatchDerivativeTester[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(BatchDerivativeTester::addRefs)
        .toArray((x) -> new BatchDerivativeTester[x][]);
  }

  public ToleranceStatistics testLearning(@Nonnull Layer component, @Nonnull IOPair IOPair,
                                          ToleranceStatistics statistics) {
    final ToleranceStatistics prev = statistics;
    statistics = com.simiacryptus.ref.wrappers.RefIntStream.range(0, component.state().size()).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null
          : measureLearningGradient(component, i, IOPair.getOutputPrototype(), IOPair.getInputPrototype());
      @Nonnull final Tensor implementedGradient = getLearningGradient(component, i, IOPair.getOutputPrototype(),
          IOPair.getInputPrototype());
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
            log.info(String.format("Weights: %s", new Tensor(component.state().get(i)).prettyPrint()));
            log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
            log.info(
                String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
              log.info(
                  String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              log.info(String.format("Gradient Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
              log.info(String.format("Error Statistics: %s",
                  new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
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
          log.info(String.format("Gradient Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
          log.info(String.format("Error Statistics: %s",
              new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
        }
        throw e;
      }

    }).reduce((a, b) -> a.combine(b)).map(x -> x.combine(prev)).orElse(prev);
    return statistics;
  }

  public ToleranceStatistics testFeedback(@Nonnull Layer component, @Nonnull IOPair IOPair,
                                          ToleranceStatistics statistics) {
    statistics = statistics
        .combine(com.simiacryptus.ref.wrappers.RefIntStream.range(0, IOPair.getInputPrototype().length).mapToObj(i -> {
          @Nullable final Tensor measuredGradient = !verify ? null
              : measureFeedbackGradient(component, i, IOPair.getOutputPrototype(), IOPair.getInputPrototype());
          @Nonnull final Tensor implementedGradient = getFeedbackGradient(component, i, IOPair.getOutputPrototype(),
              IOPair.getInputPrototype());
          try {
            final ToleranceStatistics result = com.simiacryptus.ref.wrappers.RefIntStream
                .range(0, null == measuredGradient ? 0 : measuredGradient.length()).mapToObj(i1 -> {
                  return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                      implementedGradient.getData()[i1]);
                }).reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());

            if (!(result.absoluteTol.getMax() < tolerance))
              throw new AssertionError(result.toString());
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
                log.info(
                    String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                log.info(
                    String.format("Feedback Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
                log.info(String.format("Error Statistics: %s",
                    new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
              }
            }
            return result;
          } catch (@Nonnull final Throwable e) {
            //log.info(String.format("Component: %s", component));
            log.info(String.format("Feedback for input %s", i));
            log.info(String.format("Inputs Values: %s", IOPair.getInputPrototype()[i].prettyPrint()));
            log.info(String.format("Value Statistics: %s",
                new ScalarStatistics().add(IOPair.getInputPrototype()[i].getData())));
            log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
            log.info(
                String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured: %s", measuredGradient.prettyPrint()));
              log.info(
                  String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              log.info(String.format("Feedback Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
              log.info(String.format("Error Statistics: %s",
                  new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
            }
            throw e;
          }
        }).reduce((a, b) -> a.combine(b)).get());
    return statistics;
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    log.h1("Differential Validation");
    @Nonnull
    IOPair ioPair = new IOPair(component, inputPrototype[0], BatchDerivativeTester.this).invoke();

    if (verbose) {
      log.run(() -> {
        BatchDerivativeTester.log.info(String.format("Inputs: %s", com.simiacryptus.ref.wrappers.RefArrays
            .stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get()));
        BatchDerivativeTester.log
            .info(String.format("Inputs Statistics: %s", com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype)
                .map(x -> new ScalarStatistics().add(x.getData()).toString()).reduce((a, b) -> a + ",\n" + b).get()));
        BatchDerivativeTester.log.info(String.format("Output: %s", ioPair.getOutputPrototype().prettyPrint()));
        BatchDerivativeTester.log.info(
            String.format("Outputs Statistics: %s", new ScalarStatistics().add(ioPair.getOutputPrototype().getData())));
      });
    }

    ToleranceStatistics _statistics = new ToleranceStatistics();

    if (isTestFeedback()) {
      log.h2("Feedback Validation");
      log.p(
          "We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.eval(() -> {
        return testFeedback(component, ioPair, statistics);
      });
    }
    if (isTestLearning()) {
      log.h2("Learning Validation");
      log.p(
          "We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.eval(() -> {
        return testLearning(component, ioPair, statistics);
      });
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
    log.run(() -> {
      testFrozen(component, ioPair.getInputPrototype());
      testUnFrozen(component, ioPair.getInputPrototype());
    });

    return _statistics;
  }

  public void testFrozen(@Nonnull final Layer component, @Nonnull final Tensor[] inputPrototype) {
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().freeze();
    @Nullable final Result eval = frozen.eval(new Result(new TensorArray(inputPrototype),
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

    });
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList tensorList = eval.getData().copy();
    eval.accumulate(buffer, tensorList);
    final com.simiacryptus.ref.wrappers.RefList<Delta<UUID>> deltas = component.state().stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    if (!deltas.isEmpty() && !component.state().isEmpty()) {
      throw new AssertionError("Frozen component listed in evalInputDelta. Deltas: " + deltas);
    }
    final int inElements = com.simiacryptus.ref.wrappers.RefArrays.stream(inputPrototype).mapToInt(x -> x.length())
        .sum();
    if (!reachedInputFeedback.get() && 0 < inElements) {
      throw new RuntimeException("Frozen component did not pass input backwards");
    }
  }

  public void testUnFrozen(@Nonnull final Layer component, final Tensor[] inputPrototype) {
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().setFrozen(false);
    @Nullable final Result eval = frozen.eval(new Result(new TensorArray(inputPrototype),
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

    });
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList data = eval.getData();
    eval.accumulate(buffer, data);
    @Nullable final com.simiacryptus.ref.wrappers.RefList<double[]> stateList = frozen.state();
    final com.simiacryptus.ref.wrappers.RefList<Delta<UUID>> deltas = stateList.stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      throw new AssertionError("Nonfrozen component not listed in evalInputDelta. Deltas: " + deltas);
    }
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
    final Tensor inputTensor = inputPrototype[inputIndex];
    final int inputDims = inputTensor.length();
    @Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor());
      @Nonnull final Result copyInput = new Result(new TensorArray(inputPrototype),
          (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
            @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
            if (!com.simiacryptus.ref.wrappers.RefArrays.equals(inputTensor.getDimensions(),
                data.get(inputIndex).getDimensions())) {
              throw new AssertionError();
            }
            for (int i = 0; i < inputDims; i++) {
              gradientBuffer.set(new int[]{i, j_}, data.get(inputIndex).getData()[i]);
            }
            buffer.get(inputKey.getId(), new double[gradientBuffer.length()]).addInPlace(gradientBuffer.getData());
          }) {

        @Override
        public boolean isAlive() {
          return true;
        }

        public @SuppressWarnings("unused")
        void _free() {
        }

      };
      @Nullable final Result eval = component.eval(copyInput);
      @Nonnull final DeltaSet<UUID> xxx = new DeltaSet<UUID>();
      @Nonnull
      TensorArray tensorArray = new TensorArray(eval.getData().stream().map(x -> {
        return x.set(j_, 1);
      }).toArray(i -> new Tensor[i]));
      eval.accumulate(xxx, tensorArray);
      final Delta<UUID> inputDelta = xxx.getMap().get(inputKey);
      if (null != inputDelta) {
        result.addInPlace(new Tensor(inputDelta.getDelta(), result.getDimensions()));
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
      @Nonnull final Tensor data = new Tensor(outputPrototype.getDimensions()).set((k) -> k == j_ ? 1 : 0);
      @Nullable final Result eval = component.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype}));
      eval.getData().get(0);
      @Nonnull
      TensorArray tensorArray = new TensorArray(data);
      eval.accumulate(buffer, tensorArray);
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
    @Nullable final Tensor baseOutput = component.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype}))
        .getData().get(0);
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].length(); i++) {
      @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      @Nonnull final Tensor[] copyInput = com.simiacryptus.ref.wrappers.RefArrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      @Nullable final Tensor evalProbe = component.eval(ConstantResult.singleResultArray(new Tensor[][]{copyInput})).getData()
          .get(0);
      @Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.length(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }

  @Nonnull
  private Tensor measureLearningGradient(@Nonnull final Layer component, final int layerNum,
                                         @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());

    @Nullable final Tensor baseOutput = component.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype}))
        .getData().get(0);

    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      copy.state().get(layerNum)[i] += probeSize;

      @Nullable final Tensor evalProbe = copy.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype})).getData()
          .get(0);

      @Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return gradient;
  }

  private static @com.simiacryptus.ref.lang.RefAware
  class IOPair extends ReferenceCountingBase {
    private final Layer component;
    private final Tensor tensor;
    private final BatchDerivativeTester parent;
    private Tensor[] inputPrototype;
    private Tensor outputPrototype;

    public IOPair(Layer component, Tensor tensor, BatchDerivativeTester parent) {
      this.component = component;
      this.tensor = tensor;
      this.parent = parent;
    }

    public Tensor[] getInputPrototype() {
      return inputPrototype;
    }

    public Tensor getOutputPrototype() {
      return outputPrototype;
    }

    public static @SuppressWarnings("unused")
    IOPair[] addRefs(IOPair[] array) {
      if (array == null)
        return null;
      return java.util.Arrays.stream(array).filter((x) -> x != null).map(IOPair::addRef).toArray((x) -> new IOPair[x]);
    }

    @Nonnull
    public IOPair invoke() {
      inputPrototype = com.simiacryptus.ref.wrappers.RefIntStream.range(0, parent.batches).mapToObj(i -> tensor.copy())
          .toArray(j -> new Tensor[j]);
      outputPrototype = SimpleEval.run(component, inputPrototype[0]).getOutput();
      return this;
    }

    public @SuppressWarnings("unused")
    void _free() {
    }

    public @Override
    @SuppressWarnings("unused")
    IOPair addRef() {
      return (IOPair) super.addRef();
    }
  }
}
