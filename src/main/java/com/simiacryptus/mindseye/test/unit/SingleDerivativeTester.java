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
    return this.addRef();
  }

  public boolean isTestLearning() {
    return testLearning;
  }

  @Nonnull
  public SingleDerivativeTester setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this.addRef();
  }

  public boolean isVerbose() {
    return verbose;
  }

  @Nonnull
  public SingleDerivativeTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this.addRef();
  }

  public boolean isVerify() {
    return verify;
  }

  @Nonnull
  public SingleDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this.addRef();
  }

  public static @SuppressWarnings("unused")
  SingleDerivativeTester[] addRefs(SingleDerivativeTester[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SingleDerivativeTester::addRef)
        .toArray((x) -> new SingleDerivativeTester[x]);
  }

  public static @SuppressWarnings("unused")
  SingleDerivativeTester[][] addRefs(SingleDerivativeTester[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SingleDerivativeTester::addRefs)
        .toArray((x) -> new SingleDerivativeTester[x][]);
  }

  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, @Nonnull final Layer component,
                                  @Nonnull final Tensor... inputPrototype) {
    output.h1("Differential Validation");
    ToleranceStatistics _statistics = new ToleranceStatistics();
    SimpleEval temp_00_0023 = SimpleEval.run(
        component == null ? null : component.addRef(), Tensor.addRefs(inputPrototype));
    final Tensor outputPrototype = temp_00_0023.getOutput();
    if (null != temp_00_0023)
      temp_00_0023.freeRef();
    {
      if (verbose) {
        output.run(RefUtil.wrapInterface(() -> {
              log.info(RefString.format("Inputs: %s",
                  RefArrays.stream(Tensor.addRefs(inputPrototype)).map(t -> {
                    String temp_00_0003 = t.prettyPrint();
                    if (null != t)
                      t.freeRef();
                    return temp_00_0003;
                  }).reduce((a, b) -> a + ",\n" + b).orElse("")));
              log.info(RefString.format("Inputs Statistics: %s",
                  RefArrays.stream(Tensor.addRefs(inputPrototype)).map(x -> {
                    String temp_00_0004 = new ScalarStatistics().add(x.getData()).toString();
                    if (null != x)
                      x.freeRef();
                    return temp_00_0004;
                  }).reduce((a, b) -> a + ",\n" + b).orElse("")));
              log.info(RefString.format("Output: %s", null == outputPrototype ? null : outputPrototype.prettyPrint()));
              log.info(RefString.format("Outputs Statistics: %s", new ScalarStatistics().add(outputPrototype.getData())));
            }, outputPrototype == null ? null : outputPrototype.addRef(),
            Tensor.addRefs(inputPrototype)));
      }
      if (isTestFeedback()) {
        output.h2("Feedback Validation");
        output.p(
            "We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
        final ToleranceStatistics statistics = _statistics;
        _statistics = output.eval(RefUtil.wrapInterface(
            () -> {
              return testFeedback(statistics, component == null ? null : component.addRef(),
                  Tensor.addRefs(inputPrototype),
                  outputPrototype == null ? null : outputPrototype.addRef());
            }, outputPrototype == null ? null : outputPrototype.addRef(),
            Tensor.addRefs(inputPrototype),
            component == null ? null : component.addRef()));
      }
      if (isTestLearning()) {
        output.h2("Learning Validation");
        output.p(
            "We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
        final ToleranceStatistics statistics = _statistics;
        _statistics = output.eval(RefUtil.wrapInterface(
            () -> {
              return testLearning(statistics, component == null ? null : component.addRef(),
                  Tensor.addRefs(inputPrototype),
                  outputPrototype == null ? null : outputPrototype.addRef());
            }, outputPrototype == null ? null : outputPrototype.addRef(),
            Tensor.addRefs(inputPrototype),
            component == null ? null : component.addRef()));
      }
    }
    if (null != outputPrototype)
      outputPrototype.freeRef();
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
      testFrozen(component == null ? null : component.addRef(),
          Tensor.addRefs(inputPrototype));
      testUnFrozen(component == null ? null : component.addRef(),
          Tensor.addRefs(inputPrototype));
    }, Tensor.addRefs(inputPrototype), component == null ? null : component));

    ReferenceCounting.freeRefs(inputPrototype);
    return _statistics;
  }

  public ToleranceStatistics testLearning(@Nonnull ToleranceStatistics prev, @Nonnull Layer component,
                                          Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    RefList<double[]> temp_00_0024 = component.state();
    ToleranceStatistics temp_00_0020 = RefIntStream.range(0, temp_00_0024.size())
        .mapToObj(RefUtil.wrapInterface(
            (IntFunction<ToleranceStatistics>) i -> {
              Tensor temp_00_0025 = measureLearningGradient(
                  component == null ? null : component.addRef(), i,
                  outputPrototype == null ? null : outputPrototype.addRef(),
                  Tensor.addRefs(inputPrototype));
              @Nullable final Tensor measuredGradient = !verify ? null : temp_00_0025.addRef();
              if (null != temp_00_0025)
                temp_00_0025.freeRef();
              @Nonnull final Tensor implementedGradient = getLearningGradient(component == null ? null : component.addRef(), i,
                  outputPrototype == null ? null : outputPrototype.addRef(),
                  Tensor.addRefs(inputPrototype));
              @Nonnull
              Tensor difference = measuredGradient
                  .minus(implementedGradient == null ? null : implementedGradient.addRef());
              try {
                final ToleranceStatistics result = RefIntStream
                    .range(0, null == measuredGradient ? 0 : measuredGradient.length())
                    .mapToObj(RefUtil.wrapInterface(
                        (IntFunction<ToleranceStatistics>) i1 -> {
                          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                              implementedGradient.getData()[i1]);
                        }, implementedGradient == null ? null : implementedGradient.addRef(),
                        measuredGradient == null ? null : measuredGradient.addRef()))
                    .reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
                if (!(result.absoluteTol.getMax() < tolerance)) {
                  if (null != measuredGradient)
                    measuredGradient.freeRef();
                  implementedGradient.freeRef();
                  difference.freeRef();
                  throw new AssertionError(result.toString());
                } else {
                  //log.info(String.format("Component: %s", component));
                  if (verbose) {

                    log.info(RefString.format("Learning Gradient for weight setByCoord %s", i));
                    RefList<double[]> temp_00_0026 = component.state();
                    log.info(RefString.format("Weights: %s", Tensor.prettyPrint(temp_00_0026.get(i))));
                    if (null != temp_00_0026)
                      temp_00_0026.freeRef();
                    log.info(RefString.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
                    log.info(RefString.format("Implemented Statistics: %s",
                        new ScalarStatistics().add(implementedGradient.getData())));
                    if (null != measuredGradient) {
                      log.info(RefString.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
                      log.info(RefString.format("Measured Statistics: %s",
                          new ScalarStatistics().add(measuredGradient.getData())));
                      log.info(RefString.format("Gradient Error: %s", difference.prettyPrint()));
                      log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                    }
                  }
                  if (null != measuredGradient)
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
                if (null != measuredGradient) {
                  log.info(RefString.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
                  log.info(
                      RefString.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                  log.info(RefString.format("Gradient Error: %s", difference.prettyPrint()));
                  log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                }
                throw e;
              } finally {
              }

            }, outputPrototype == null ? null : outputPrototype, component == null ? null : component,
            Tensor.addRefs(inputPrototype)))
        .reduce((a, b) -> a.combine(b)).map(x -> x.combine(prev)).orElse(prev);
    if (null != temp_00_0024)
      temp_00_0024.freeRef();
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    return temp_00_0020;
  }

  @Nonnull
  public ToleranceStatistics testFeedback(@Nonnull ToleranceStatistics statistics, @Nonnull Layer component,
                                          @Nonnull Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    Optional<ToleranceStatistics> optional = RefIntStream.range(0, inputPrototype.length)
        .mapToObj(RefUtil.wrapInterface(
            (IntFunction<ToleranceStatistics>) i -> {
              Tensor temp_00_0027 = measureFeedbackGradient(
                  component == null ? null : component.addRef(), i,
                  outputPrototype == null ? null : outputPrototype.addRef(),
                  Tensor.addRefs(inputPrototype));
              @Nullable final Tensor measuredGradient = !verify ? null : temp_00_0027.addRef();
              if (null != temp_00_0027)
                temp_00_0027.freeRef();
              @Nonnull final Tensor implementedGradient = getFeedbackGradient(component == null ? null : component.addRef(), i,
                  outputPrototype == null ? null : outputPrototype.addRef(),
                  Tensor.addRefs(inputPrototype));
              Tensor maskedGradient = implementedGradient.mapCoords(RefUtil.wrapInterface(
                  c -> Double
                      .isNaN(measuredGradient.get(c.getCoords())) ? Double.NaN : implementedGradient.get(c),
                  implementedGradient == null ? null : implementedGradient.addRef(),
                  measuredGradient == null ? null : measuredGradient.addRef()));
              @Nonnull
              Tensor difference = measuredGradient.minus(maskedGradient == null ? null : maskedGradient.addRef());
              try {
                final ToleranceStatistics result = RefIntStream
                    .range(0, null == measuredGradient ? 0 : measuredGradient.length())
                    .mapToObj(RefUtil.wrapInterface(
                        (IntFunction<ToleranceStatistics>) i1 -> {
                          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1],
                              maskedGradient.getData()[i1]);
                        }, maskedGradient == null ? null : maskedGradient.addRef(),
                        measuredGradient == null ? null : measuredGradient.addRef()))
                    .reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());

                if (!(result.absoluteTol.getMax() < tolerance)) {
                  if (null != measuredGradient)
                    measuredGradient.freeRef();
                  implementedGradient.freeRef();
                  if (null != maskedGradient)
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
                  if (null != measuredGradient) {
                    log.info(RefString.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
                    log.info(RefString.format("Measured Statistics: %s",
                        new ScalarStatistics().add(measuredGradient.getData())));
                    log.info(RefString.format("Feedback Error: %s", difference.prettyPrint()));
                    log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                  }
                }

                if (null != measuredGradient)
                  measuredGradient.freeRef();
                implementedGradient.freeRef();
                if (null != maskedGradient)
                  maskedGradient.freeRef();
                difference.freeRef();
                return result;
              } catch (@Nonnull final Throwable e) {
                //log.info(String.format("Component: %s", component));
                log.info(RefString.format("Feedback for input %s", i));
                log.info(RefString.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
                log.info(
                    RefString.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
                if (!implementedGradient.isFinalized()) {
                  log.info(RefString.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
                  log.info(RefString.format("Implemented Statistics: %s",
                      new ScalarStatistics().add(implementedGradient.getData())));
                }
                if (null != measuredGradient) {
                  log.info(RefString.format("Measured: %s", measuredGradient.prettyPrint()));
                  log.info(
                      RefString.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
                }
                if (!difference.isFinalized()) {
                  log.info(RefString.format("Feedback Error: %s", difference.prettyPrint()));
                  log.info(RefString.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
                }
                throw e;
              } finally {
              }
            }, component == null ? null : component, Tensor.addRefs(inputPrototype),
            outputPrototype == null ? null : outputPrototype))
        .reduce((a, b) -> a.combine(b));
    ReferenceCounting.freeRefs(inputPrototype);
    if (!optional.isPresent())
      return statistics;
    return statistics.combine(optional.orElse(null));
  }

  public void testFrozen(@Nonnull final Layer component, @Nonnull Tensor[] inputPrototype) {
    final int inElements = RefArrays.stream(Tensor.addRefs(inputPrototype))
        .mapToInt(x -> {
          int temp_00_0005 = x.length();
          if (null != x)
            x.freeRef();
          return temp_00_0005;
        }).sum();
    inputPrototype = RefArrays.stream(Tensor.addRefs(inputPrototype)).map(tensor -> {
      Tensor temp_00_0006 = tensor.copy();
      if (null != tensor)
        tensor.freeRef();
      return temp_00_0006;
    }).toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_00_0028 = component.copy();
    @Nonnull final Layer frozen = temp_00_0028.freeze();
    if (null != temp_00_0028)
      temp_00_0028.freeRef();
    RefList<TensorArray> inputCopies = RefArrays.stream(Tensor.addRefs(inputPrototype))
        .map(data -> {
          TensorArray temp_00_0007 = new TensorArray(
              data == null ? null : data.addRef());
          if (null != data)
            data.freeRef();
          return temp_00_0007;
        }).collect(RefCollectors.toList());
    ReferenceCounting.freeRefs(inputPrototype);
    Result[] input = inputCopies.stream().map((tensorArray) -> {
      try {
        return new Result(tensorArray, new Result.Accumulator() {
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

        };
      } finally {
        if (null != tensorArray)
          tensorArray.freeRef();
      }
    }).toArray(i -> new Result[i]);
    if (null != inputCopies)
      inputCopies.freeRef();
    @Nullable final Result eval;
    eval = frozen.eval(Result.addRefs(input));
    if (null != input)
      ReferenceCounting.freeRefs(input);
    frozen.freeRef();
    @Nonnull final DeltaSet<UUID> buffer;
    TensorList tensorList;
    TensorList evalData = eval.getData();
    {
      buffer = new DeltaSet<UUID>();
      tensorList = evalData.copy();
      eval.accumulate(buffer == null ? null : buffer.addRef(), tensorList == null ? null : tensorList.addRef());
    }
    if (null != evalData)
      evalData.freeRef();
    if (null != tensorList)
      tensorList.freeRef();
    if (null != eval)
      eval.freeRef();
    RefList<double[]> temp_00_0029 = component.state();
    final RefList<Delta<UUID>> deltas = temp_00_0029.stream().map(RefUtil.wrapInterface(
        (Function<? super double[], ? extends Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_00_0031 = buffer.stream()
              .filter(x -> {
                boolean temp_00_0009 = x.target == doubles;
                if (null != x)
                  x.freeRef();
                return temp_00_0009;
              }).findFirst();
          Delta<UUID> temp_00_0030 = temp_00_0031.orElse(null);
          if (null != temp_00_0031)
            RefUtil.freeRef(temp_00_0031);
          return temp_00_0030;
        }, buffer == null ? null : buffer)).filter(x -> {
      boolean temp_00_0010 = x != null;
      if (null != x)
        x.freeRef();
      return temp_00_0010;
    }).collect(RefCollectors.toList());
    if (null != temp_00_0029)
      temp_00_0029.freeRef();
    RefList<double[]> temp_00_0032 = component.state();
    if (!deltas.isEmpty() && !temp_00_0032.isEmpty()) {
      AssertionError temp_00_0011 = new AssertionError(
          "Frozen component listed in evalInputDelta. Deltas: " + deltas);
      if (null != deltas)
        deltas.freeRef();
      component.freeRef();
      throw temp_00_0011;
    }
    if (null != temp_00_0032)
      temp_00_0032.freeRef();
    component.freeRef();
    if (null != deltas)
      deltas.freeRef();
    if (!reachedInputFeedback.get() && 0 < inElements) {
      throw new RuntimeException("Frozen component did not pass input backwards");
    }
  }

  public void testUnFrozen(@Nonnull final Layer component, Tensor[] inputPrototype) {
    inputPrototype = RefArrays.stream(Tensor.addRefs(inputPrototype)).map(tensor -> {
      Tensor temp_00_0012 = tensor.copy();
      if (null != tensor)
        tensor.freeRef();
      return temp_00_0012;
    }).toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    Layer temp_00_0033 = component.copy();
    @Nonnull final Layer frozen = temp_00_0033.setFrozen(false);
    if (null != temp_00_0033)
      temp_00_0033.freeRef();
    component.freeRef();
    RefList<TensorArray> inputCopies = RefArrays.stream(Tensor.addRefs(inputPrototype))
        .map(data -> {
          TensorArray temp_00_0013 = new TensorArray(
              data == null ? null : data.addRef());
          if (null != data)
            data.freeRef();
          return temp_00_0013;
        }).collect(RefCollectors.toList());
    Result[] inputs = inputCopies.stream().map(tensor -> {
      try {
        return new Result(tensor, new Result.Accumulator() {
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
        };
      } finally {
        if (null != tensor)
          tensor.freeRef();
      }
    }).toArray(i -> new Result[i]);
    if (null != inputCopies)
      inputCopies.freeRef();
    @Nullable final Result eval;
    eval = frozen.eval(Result.addRefs(inputs));
    if (null != inputs)
      ReferenceCounting.freeRefs(inputs);
    @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
    TensorList tensorList = eval.getData();
    eval.accumulate(buffer == null ? null : buffer.addRef(), tensorList == null ? null : tensorList.addRef());
    if (null != tensorList)
      tensorList.freeRef();
    if (null != eval)
      eval.freeRef();
    @Nullable final RefList<double[]> stateList = frozen.state();
    frozen.freeRef();
    final RefList<Delta<UUID>> deltas = stateList.stream().map(RefUtil.wrapInterface(
        (Function<? super double[], ? extends Delta<UUID>>) doubles -> {
          Optional<Delta<UUID>> temp_00_0035 = buffer.stream()
              .filter(x -> {
                boolean temp_00_0015 = x.target == doubles;
                if (null != x)
                  x.freeRef();
                return temp_00_0015;
              }).findFirst();
          Delta<UUID> temp_00_0034 = temp_00_0035.orElse(null);
          if (null != temp_00_0035)
            RefUtil.freeRef(temp_00_0035);
          return temp_00_0034;
        }, buffer == null ? null : buffer)).filter(x -> {
      boolean temp_00_0016 = x != null;
      if (null != x)
        x.freeRef();
      return temp_00_0016;
    }).collect(RefCollectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      if (null != stateList)
        stateList.freeRef();
      AssertionError temp_00_0017 = new AssertionError(
          "Nonfrozen component not listed in evalInputDelta. Deltas: " + deltas);
      if (null != deltas)
        deltas.freeRef();
      if (null != inputPrototype)
        ReferenceCounting.freeRefs(inputPrototype);
      throw temp_00_0017;
    }
    if (null != deltas)
      deltas.freeRef();
    if (null != stateList)
      stateList.freeRef();
    if (!reachedInputFeedback.get() && inputPrototype.length != 0) {
      if (null != inputPrototype)
        ReferenceCounting.freeRefs(inputPrototype);
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
    if (null != inputPrototype)
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

  public @Override
  @SuppressWarnings("unused")
  SingleDerivativeTester addRef() {
    return (SingleDerivativeTester) super.addRef();
  }

  protected void measureFeedback(@Nonnull Layer component, int inputIndex, Tensor baseOutput,
                                 @Nonnull Tensor[] inputPrototype, Tensor measuredGradient, int probeIndex) {
    @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
    inputProbe.add(probeIndex, probeSize * 1);
    @Nonnull final Tensor[] copyInput = RefArrays.copyOf(Tensor.addRefs(inputPrototype),
        inputPrototype.length);
    ReferenceCounting.freeRefs(inputPrototype);
    {
      Tensor temp_00_0001 = inputProbe == null ? null : inputProbe.addRef();
      if (null != copyInput[inputIndex])
        copyInput[inputIndex].freeRef();
      copyInput[inputIndex] = temp_00_0001 == null ? null : temp_00_0001.addRef();
      if (null != temp_00_0001)
        temp_00_0001.freeRef();
    }
    inputProbe.freeRef();
    Result[] input1 = ConstantResult
        .batchResultArray(new Tensor[][]{Tensor.addRefs(copyInput)});
    ReferenceCounting.freeRefs(copyInput);
    try {
      Result temp_00_0036 = component
          .eval(Result.addRefs(input1));
      TensorList temp_00_0037 = temp_00_0036.getData();
      @Nullable final Tensor evalProbe = temp_00_0037.get(0);
      if (null != temp_00_0037)
        temp_00_0037.freeRef();
      if (null != temp_00_0036)
        temp_00_0036.freeRef();
      Tensor temp_00_0038 = evalProbe
          .minus(baseOutput == null ? null : baseOutput.addRef());
      @Nonnull final Tensor delta = temp_00_0038.scaleInPlace(1. / probeSize);
      if (null != temp_00_0038)
        temp_00_0038.freeRef();
      if (null != evalProbe)
        evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        measuredGradient.set(new int[]{probeIndex, j}, delta.getData()[j]);
      }
      delta.freeRef();
    } finally {
      for (@Nonnull
          Result result : input1) {
        RefUtil.freeRef(result.getData());
      }

    }
    if (null != measuredGradient)
      measuredGradient.freeRef();
    if (null != baseOutput)
      baseOutput.freeRef();
    component.freeRef();
    if (null != input1)
      ReferenceCounting.freeRefs(input1);
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
      final Result[] copyInput = RefArrays.stream(Tensor.addRefs(inputPrototype))
          .map(x -> {
            try {
              return new Result(new TensorArray(x == null ? null : x.addRef()), new Result.Accumulator() {
                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList data) {
                }
              }) {

                @Override
                public boolean isAlive() {
                  return false;
                }

                public @SuppressWarnings("unused")
                void _free() {
                }

              };
            } finally {
              if (null != x)
                x.freeRef();
            }
          }).toArray(i -> new Result[i]);
      RefUtil.freeRef(copyInput[inputIndex].getData());
      double[] target = new double[inputDims * outputPrototype.length()];
      {
        Result temp_00_0002 = new Result(
            new TensorArray(inputTensor == null ? null : inputTensor.addRef()), new Result.Accumulator() {
          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList data) {
            {
              if (1 != data.length())
                throw new AssertionError();
              if (data.length() != 1)
                throw new AssertionError();
              @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
              if (!RefArrays.equals(inputTensor.getDimensions(), data.getDimensions())) {
                throw new AssertionError();
              }
              RefIntStream.range(0, data.length()).forEach(dataIndex -> {
                for (int i = 0; i < inputDims; i++) {
                  @Nullable
                  Tensor tensor = data.get(dataIndex);
                  gradientBuffer.set(new int[]{i, j_}, tensor.getData()[i]);
                }
              });
              buffer.get(inputKey.getId(), target).addInPlace(gradientBuffer.getData());
            }
          }
        }) {

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
          }
        };
        if (null != copyInput[inputIndex])
          copyInput[inputIndex].freeRef();
        copyInput[inputIndex] = temp_00_0002 == null ? null : temp_00_0002.addRef();
        if (null != temp_00_0002)
          temp_00_0002.freeRef();
      }
      @Nullable final Result eval;
      try {
        eval = component.eval(Result.addRefs(copyInput));
      } finally {
        for (@Nonnull
            Result nnResult : copyInput) {
          RefUtil.freeRef(nnResult.getData());
        }
      }
      if (null != copyInput)
        ReferenceCounting.freeRefs(copyInput);
      @Nonnull final DeltaSet<UUID> deltaSet = new DeltaSet<UUID>();
      Tensor temp_00_0021 = new Tensor(outputPrototype.getDimensions());
      @Nonnull
      TensorArray tensorArray = new TensorArray(temp_00_0021.set(j, 1));
      if (null != temp_00_0021)
        temp_00_0021.freeRef();
      try {
        eval.accumulate(deltaSet == null ? null : deltaSet.addRef(), tensorArray == null ? null : tensorArray.addRef());
        RefMap<UUID, Delta<UUID>> map = deltaSet.getMap();
        final Delta<UUID> inputDelta = map.get(inputKey.getId());
        if (null != map)
          map.freeRef();
        if (null != inputDelta) {
          @Nonnull
          Tensor tensor = new Tensor(inputDelta.getDelta(), result.getDimensions());
          result.addInPlace(tensor == null ? null : tensor);
        }
        if (null != inputDelta)
          inputDelta.freeRef();
      } finally {
        RefUtil.freeRef(eval.getData());
      }
      tensorArray.freeRef();
      deltaSet.freeRef();
      if (null != eval)
        eval.freeRef();
      inputKey.freeRef();
    }
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
    RefList<double[]> temp_00_0039 = component.state();
    final double[] stateArray = temp_00_0039.get(layerNum);
    if (null != temp_00_0039)
      temp_00_0039.freeRef();
    final int stateLen = stateArray.length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final DeltaSet<UUID> buffer = new DeltaSet<UUID>();
      Result[] array = ConstantResult
          .batchResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)});
      @Nullable final Result eval = component.eval(Result.addRefs(array));
      for (@Nonnull
          Result result : array) {
        RefUtil.freeRef(result.getData());
      }
      if (null != array)
        ReferenceCounting.freeRefs(array);
      Tensor temp_00_0022 = new Tensor(outputPrototype.getDimensions());
      @Nonnull
      TensorArray tensorArray = new TensorArray(temp_00_0022.set((k) -> k == j_ ? 1 : 0));
      if (null != temp_00_0022)
        temp_00_0022.freeRef();
      eval.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
      RefUtil.freeRef(eval.getData());
      if (null != eval)
        eval.freeRef();
      RefMap<UUID, Delta<UUID>> temp_00_0040 = buffer
          .getMap();
      RefCollection<Delta<UUID>> temp_00_0041 = temp_00_0040
          .values();
      Optional<Delta<UUID>> temp_00_0042 = temp_00_0041.stream()
          .filter(x -> {
            boolean temp_00_0019 = x.target == stateArray;
            if (null != x)
              x.freeRef();
            return temp_00_0019;
          }).findFirst();
      final DoubleBuffer<UUID> deltaFlushBuffer = temp_00_0042.orElse(null);
      if (null != temp_00_0042)
        RefUtil.freeRef(temp_00_0042);
      if (null != temp_00_0041)
        temp_00_0041.freeRef();
      if (null != temp_00_0040)
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
    @Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].length(), outputPrototype.length());
    Result[] input0 = ConstantResult
        .batchResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)});
    Result temp_00_0043 = component
        .eval(Result.addRefs(input0));
    TensorList temp_00_0044 = temp_00_0043.getData();
    @Nullable final Tensor baseOutput = temp_00_0044.get(0);
    if (null != temp_00_0044)
      temp_00_0044.freeRef();
    if (null != temp_00_0043)
      temp_00_0043.freeRef();
    for (@Nonnull
        Result result : input0) {
      RefUtil.freeRef(result.getData());
    }
    if (null != input0)
      ReferenceCounting.freeRefs(input0);
    outputPrototype.set(baseOutput == null ? null : baseOutput.addRef());
    outputPrototype.freeRef();
    for (int probeIndex = 0; probeIndex < inputPrototype[inputIndex].length(); probeIndex++) {
      measureFeedback(component == null ? null : component.addRef(), inputIndex,
          baseOutput == null ? null : baseOutput.addRef(),
          Tensor.addRefs(inputPrototype),
          measuredGradient == null ? null : measuredGradient.addRef(), probeIndex);
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
    RefList<double[]> temp_00_0045 = component.state();
    final int stateLen = temp_00_0045.get(layerNum).length;
    if (null != temp_00_0045)
      temp_00_0045.freeRef();
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());

    outputPrototype.freeRef();
    Result[] input2 = ConstantResult
        .batchResultArray(new Tensor[][]{Tensor.addRefs(inputPrototype)});
    if (null != inputPrototype)
      ReferenceCounting.freeRefs(inputPrototype);
    Result temp_00_0046 = component
        .eval(Result.addRefs(input2));
    TensorList temp_00_0047 = temp_00_0046.getData();
    @Nullable final Tensor baseOutput = temp_00_0047.get(0);

    if (null != temp_00_0047)
      temp_00_0047.freeRef();
    if (null != temp_00_0046)
      temp_00_0046.freeRef();
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      RefList<double[]> temp_00_0048 = copy.state();
      temp_00_0048.get(layerNum)[i] += probeSize;
      if (null != temp_00_0048)
        temp_00_0048.freeRef();
      Result temp_00_0049 = copy
          .eval(Result.addRefs(input2));
      TensorList temp_00_0050 = temp_00_0049.getData();
      @Nullable final Tensor evalProbe = temp_00_0050.get(0);
      if (null != temp_00_0050)
        temp_00_0050.freeRef();
      if (null != temp_00_0049)
        temp_00_0049.freeRef();
      copy.freeRef();
      Tensor temp_00_0051 = evalProbe
          .minus(baseOutput == null ? null : baseOutput.addRef());
      @Nonnull final Tensor delta = temp_00_0051.scaleInPlace(1. / probeSize);
      if (null != temp_00_0051)
        temp_00_0051.freeRef();
      if (null != evalProbe)
        evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    component.freeRef();
    if (null != baseOutput)
      baseOutput.freeRef();
    for (@Nonnull
        Result result : input2) {
      RefUtil.freeRef(result.getData());
    }
    if (null != input2)
      ReferenceCounting.freeRefs(input2);
    return gradient;
  }
}
