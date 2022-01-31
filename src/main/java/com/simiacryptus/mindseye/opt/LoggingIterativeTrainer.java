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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.IterativeStopException;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.line.*;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

public abstract class LoggingIterativeTrainer extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(LoggingIterativeTrainer.class);
  private final RefMap<CharSequence, LineSearchStrategy> lineSearchStrategyMap = new RefHashMap<>();
  @Nullable
  private final Trainable subject;
  private AtomicInteger iterationCounter = new AtomicInteger(0);
  private int iterationsPerSample = 100;
  private Function<CharSequence, LineSearchStrategy> lineSearchFactory = s -> new ArmijoWolfeSearch();
  private int maxIterations = Integer.MAX_VALUE;
  private TrainingMonitor monitor = new TrainingMonitor();
  @Nullable
  private OrientationStrategy<?> orientation = new LBFGS();
  private double terminateThreshold;
  private Duration timeout;
  private boolean iterationSubreports = false;

  public LoggingIterativeTrainer(@Nullable final Trainable subject) {
    this.subject = subject;
    timeout = Duration.of(5, ChronoUnit.MINUTES);
    terminateThreshold = 0;
  }

  public AtomicInteger getIterationCounter() {
    return iterationCounter;
  }

  public void setIterationCounter(AtomicInteger iterationCounter) {
    this.iterationCounter = iterationCounter;
  }

  public int getIterationsPerSample() {
    return iterationsPerSample;
  }

  public void setIterationsPerSample(int iterationsPerSample) {
    this.iterationsPerSample = iterationsPerSample;
  }

  public Function<CharSequence, LineSearchStrategy> getLineSearchFactory() {
    return lineSearchFactory;
  }

  public void setLineSearchFactory(Function<CharSequence, LineSearchStrategy> lineSearchFactory) {
    this.lineSearchFactory = lineSearchFactory;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  public TrainingMonitor getMonitor() {
    return monitor;
  }

  public void setMonitor(TrainingMonitor monitor) {
    this.monitor = monitor;
  }

  @Nullable
  public OrientationStrategy<?> getOrientation() {
    return orientation == null ? null : orientation.addRef();
  }

  public void setOrientation(@Nullable OrientationStrategy<?> orientation) {
    if (null != this.orientation)
      this.orientation.freeRef();
    this.orientation = orientation;
  }

  public double getTerminateThreshold() {
    return terminateThreshold;
  }

  public void setTerminateThreshold(double terminateThreshold) {
    this.terminateThreshold = terminateThreshold;
  }

  public Duration getTimeout() {
    return timeout;
  }

  public void setTimeout(Duration timeout) {
    this.timeout = timeout;
  }

  public boolean isIterationSubreports() {
    return iterationSubreports;
  }

  public void setIterationSubreports(boolean iterationSubreports) {
    this.iterationSubreports = iterationSubreports;
  }

  @Nullable
  public PointSample measure(NotebookOutput out) {
    assert subject != null;
    final PointSample currentPoint = subject.measure(monitor);
    int size = currentPoint.delta.size();
    if (0 >= size) {
      currentPoint.freeRef();
      throw new AssertionError("Nothing to optimize");
    }
    double mean = currentPoint.getMean();
    if (!Double.isFinite(mean)) {
      if (monitor.onStepFail(new Step(currentPoint, iterationCounter.get()))) {
        monitor.log(RefString.format("Retrying iteration %s", iterationCounter.get()));
        return measure(out);
      } else {
        monitor.log(RefString.format("Optimization terminated %s", iterationCounter.get()));
        throw new IterativeStopException(Double.toString(mean));
      }
    }
    return currentPoint;
  }

  public void shuffle() {
    long seed = RefSystem.nanoTime();
    monitor.log(RefString.format("Reset training subject: " + seed));
    assert orientation != null;
    orientation.reset();
    assert subject != null;
    subject.reseed(seed);
    Layer layer = subject.getLayer();
    try {
      if (layer instanceof DAGNetwork) {
        ((DAGNetwork) layer).shuffle(seed);
      }
    } finally {
      if (null != layer)
        layer.freeRef();
    }
  }

  @RefIgnore
  public TrainingResult run(NotebookOutput out) {
    final RefAtomicReference<@Nullable PointSample> currentPointRef = new RefAtomicReference<>();
    long startTime = RefSystem.currentTimeMillis();
    final long timeoutMs = startTime + timeout.toMillis();
    AtomicLong lastIterationTime = new AtomicLong(RefSystem.nanoTime());
    shuffle();
    TrainingResult.TerminationCause terminationCause = TrainingResult.TerminationCause.Completed;
    currentPointRef.set(measure(out));
    try {
      assert isDefined(currentPointRef.get());
      mainLoop: while (timeoutMs > RefSystem.currentTimeMillis()
          && terminateThreshold < getMean(currentPointRef.get()) && maxIterations > iterationCounter.get()) {
        shuffle();
        currentPointRef.set(measure(out));
        for (int subiteration = 0; subiteration < iterationsPerSample || iterationsPerSample <= 0; subiteration++) {
          if (timeoutMs < RefSystem.currentTimeMillis()) {
            terminationCause = TrainingResult.TerminationCause.Timeout;
            break mainLoop;
          }
          if (iterationCounter.incrementAndGet() > maxIterations) {
            terminationCause = TrainingResult.TerminationCause.Completed;
            break mainLoop;
          }
          int currentIteration = iterationCounter.get();
          int stepResult;
          if (isIterationSubreports()) {
            stepResult = out.subreport("Iteration " + currentIteration, sublog -> {
              logState(sublog, currentIteration);
              return runStep(lastIterationTime, currentIteration, currentPointRef.addRef(), sublog);
            });
          } else {
            out.h3("Iteration " + currentIteration);
            logState(out, currentIteration);
            stepResult = runStep(lastIterationTime, currentIteration, currentPointRef.addRef(), out);
          }
          if (0 == stepResult) {
          } else if (1 == stepResult) {
            break;
          } else if (2 == stepResult) {
            terminationCause = TrainingResult.TerminationCause.Failed;
            break mainLoop;
          } else {
            throw new RuntimeException();
          }
        }
      }
      assert subject != null;
      Layer subjectLayer = subject.getLayer();
      if (subjectLayer instanceof DAGNetwork) {
        ((DAGNetwork) subjectLayer).clearNoise();
      }
      if (null != subjectLayer)
        subjectLayer.freeRef();
      return new TrainingResult(isDefined(currentPointRef.get()) ? getMean(currentPointRef.get()) : Double.NaN, terminationCause);
    } catch (Throwable e) {
      out.p(RefString.format("Error %s", Util.toString(e)));
      throw Util.throwException(e);
    } finally {
      out.p(RefString.format("Final threshold in iteration %s: %s (> %s) after %.3fs (< %.3fs)", iterationCounter.get(),
          isDefined(currentPointRef.get()) ? getMean(currentPointRef.get()) : null, terminateThreshold,
          (RefSystem.currentTimeMillis() - startTime) / 1000.0, timeout.toMillis() / 1000.0));
      currentPointRef.freeRef();
    }
  }

  private static boolean isDefined(Object pointSample) {
    boolean isDefined = pointSample != null;
    RefUtil.freeRef(pointSample);
    return isDefined;
  }

  public void setTimeout(int number, @Nonnull TemporalUnit units) {
    timeout = Duration.of(number, units);
  }

  public void setTimeout(int number, @Nonnull TimeUnit units) {
    setTimeout(number, Util.cvt(units));
  }

  @Nullable
  public PointSample step(@Nonnull final LineSearchCursor direction, final CharSequence directionType,
      @Nonnull final PointSample previous) {
    LineSearchStrategy lineSearchStrategy;
    if (lineSearchStrategyMap.containsKey(directionType)) {
      lineSearchStrategy = lineSearchStrategyMap.get(directionType);
    } else {
      log.info(RefString.format("Constructing line search parameters: %s", directionType));
      lineSearchStrategy = lineSearchFactory.apply(direction.getDirectionType());
      RefUtil.freeRef(lineSearchStrategyMap.put(directionType, lineSearchStrategy));
    }
    @Nonnull
    final FailsafeLineSearchCursor wrapped = new FailsafeLineSearchCursor(direction, previous, monitor);
    assert lineSearchStrategy != null;
    try {
      RefUtil.freeRef(lineSearchStrategy.step(wrapped.addRef(), monitor));
      return wrapped.getBest();
    } finally {
      wrapped.freeRef();
    }
  }

  public void _free() {
    super._free();
    if (null != orientation)
      orientation.freeRef();
    orientation = null;
    if (null != subject)
      subject.freeRef();
    lineSearchStrategyMap.freeRef();
  }

  @Nonnull
  public @Override @SuppressWarnings("unused") LoggingIterativeTrainer addRef() {
    return (LoggingIterativeTrainer) super.addRef();
  }

  protected abstract void logState(NotebookOutput sublog, int iteration);

  private static double getMean(PointSample pointSample) {
    double mean = pointSample.getMean();
    pointSample.freeRef();
    return mean;
  }

  private int runStep(AtomicLong lastIterationTime, int currentIteration,
      RefAtomicReference<@Nullable PointSample> currentPointRef, NotebookOutput out) {
    currentPointRef.set(measure(out));
    @Nullable
    final PointSample currentPoint = currentPointRef.get();
    TrainingMonitor stepMonitor = new TrainingMonitor() {
      @Override
      public void clear() {
        monitor.clear();
      }

      @Override
      public void log(String msg) {
        out.p(msg);
      }

      @Override
      public void onStepComplete(@Nullable Step currentPoint) {
        monitor.onStepComplete(currentPoint);
      }

      @Override
      public boolean onStepFail(@Nullable Step currentPoint) {
        return monitor.onStepFail(currentPoint);
      }
    };
    @Nonnull
    final TimedResult<LineSearchCursor> timedOrientation = TimedResult
        .time(RefUtil.wrapInterface((UncheckedSupplier<LineSearchCursor>) () -> {
          return orientation.orient(subject == null ? null : subject.addRef(),
              currentPoint == null ? null : currentPoint.addRef(), stepMonitor);
        }, currentPoint));
    final LineSearchCursor direction = timedOrientation.getResult();
    @Nullable
    final PointSample previous = currentPointRef.get();
    try {
      @Nonnull
      final TimedResult<PointSample> timedLineSearch = TimedResult.time(RefUtil.wrapInterface(
          (UncheckedSupplier<PointSample>) () -> step(direction.addRef(), direction.getDirectionType(),
              previous == null ? null : previous.addRef()),
          previous == null ? null : previous.addRef(), direction.addRef()));
      final long now = System.nanoTime();
      long elapsed = now - lastIterationTime.get();
      lastIterationTime.set(now);
      final CharSequence perfString = RefString.format("Total: %.4f; Orientation: %.4f; Line Search: %.4f",
          elapsed / 1e9, timedOrientation.timeNanos / 1e9, timedLineSearch.timeNanos / 1e9);
      currentPointRef.set(timedLineSearch.getResult());
      timedLineSearch.freeRef();
      assert previous != null;
      double mean = getMean(currentPointRef.get());
      stepMonitor.log(RefString.format("Fitness changed from %s to %s", previous.getMean(), mean));
      int stepResult;
      if (previous.getMean() <= mean) {
        if (previous.getMean() < mean) {
          stepMonitor.log(RefString.format("Resetting Iteration %s", perfString));
          LineSearchPoint point = direction.step(0, stepMonitor);
          assert point != null;
          currentPointRef.set(point.getPoint());
          point.freeRef();
        } else {
          stepMonitor.log(RefString.format("Static Iteration %s", perfString));
        }
        stepResult = stepResult(currentIteration, previous.addRef(), currentPointRef.get());
      } else {
        stepMonitor.log(RefString.format("Iteration %s complete. Error: %s " + perfString, currentIteration, mean));
        stepResult = 0;
      }
      if (0 == stepResult)
        stepMonitor.onStepComplete(new Step(currentPointRef.get(), currentIteration));
      return stepResult;
    } finally {
      timedOrientation.freeRef();
      previous.freeRef();
      direction.freeRef();
      currentPointRef.freeRef();
    }
  }

  private int stepResult(int currentIteration, @Nullable PointSample previous, @Nullable PointSample pointSample) {
    int stepResult;
    monitor.log(RefString.format("Iteration %s failed. Error: %s", currentIteration, pointSample.getMean()));
    monitor.log(RefString.format("Previous Error: %s -> %s", previous.getRate(), previous.getMean()));
    previous.freeRef();
    if (monitor.onStepFail(new Step(pointSample, currentIteration))) {
      monitor.log(RefString.format("Retrying iteration %s", currentIteration));
      stepResult = 1;
    } else {
      monitor.log(RefString.format("Optimization terminated %s", currentIteration));
      stepResult = 2;
    }
    return stepResult;
  }
}
