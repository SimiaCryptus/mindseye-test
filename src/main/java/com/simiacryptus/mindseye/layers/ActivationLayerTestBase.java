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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.lang.UncheckedSupplier;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;
import java.util.function.Function;

public abstract class ActivationLayerTestBase extends LayerTestBase {

  private final Layer layer;

  public ActivationLayerTestBase(final Layer layer) {
    Layer temp_03_0001 = layer == null ? null : layer.addRef();
    this.layer = temp_03_0001 == null ? null : temp_03_0001.addRef();
    if (null != temp_03_0001)
      temp_03_0001.freeRef();
    if (null != layer)
      layer.freeRef();
  }

  @Nullable
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    TrainingTester temp_03_0004 = new TrainingTester() {

      public @SuppressWarnings("unused") void _free() {
      }

      @Override
      protected Layer lossLayer() {
        return ActivationLayerTestBase.this.lossLayer();
      }
    };
    TrainingTester temp_03_0003 = temp_03_0004.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    if (null != temp_03_0004)
      temp_03_0004.freeRef();
    return temp_03_0003;
  }

  @Nonnull
  public static PlotCanvas plot(final String title, final double[][] data) {
    @Nonnull
    final PlotCanvas plot = ScatterPlot.plot(data);
    plot.setTitle(title);
    plot.setAxisLabels("x", "y");
    plot.setSize(600, 400);
    return plot;
  }

  @Nonnull
  public static PlotCanvas plot(final String title, @Nonnull final RefList<double[]> plotData,
      final Function<double[], double[]> function) {
    final double[][] data = plotData.stream().map(function).toArray(i -> new double[i][]);
    plotData.freeRef();
    return ActivationLayerTestBase.plot(title, data);
  }

  public static @SuppressWarnings("unused") ActivationLayerTestBase[] addRefs(ActivationLayerTestBase[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ActivationLayerTestBase::addRef)
        .toArray((x) -> new ActivationLayerTestBase[x]);
  }

  public static @SuppressWarnings("unused") ActivationLayerTestBase[][] addRefs(ActivationLayerTestBase[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ActivationLayerTestBase::addRefs)
        .toArray((x) -> new ActivationLayerTestBase[x][]);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][] { { 2, 3, 1 } };
  }

  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer == null ? null : layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][] { { 100, 100, 1 } };
  }

  public RefDoubleStream scan() {
    return RefIntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0);
  }

  @Override
  public void run(@NotNull final NotebookOutput log) {
    super.run(log);

    log.h3("Function Plots");
    final Layer layer = getLayer(new int[][] { { 1 } }, new Random());
    final RefList<double[]> plotData = scan().mapToObj(RefUtil.wrapInterface((DoubleFunction<? extends double[]>) x -> {
      @Nonnull
      Tensor tensor = new Tensor(x);
      @Nonnull
      final SimpleEval eval = SimpleEval.run(layer == null ? null : layer.addRef(), tensor == null ? null : tensor);
      Tensor temp_03_0005 = eval.getOutput();
      Tensor[] derivative = eval.getDerivative();
      double[] temp_03_0002 = new double[] { x, temp_03_0005.get(0), derivative[0].get(0) };
      ReferenceCounting.freeRefs(derivative);
      if (null != temp_03_0005)
        temp_03_0005.freeRef();
      eval.freeRef();
      return temp_03_0002;
    }, layer == null ? null : layer.addRef())).collect(RefCollectors.toList());

    if (null != layer)
      layer.freeRef();
    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[] { x[0], x[1] });
    }, plotData == null ? null : plotData.addRef()));

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[] { x[0], x[2] });
    }, plotData == null ? null : plotData.addRef()));
    if (null != plotData)
      plotData.freeRef();

  }

  public @SuppressWarnings("unused") void _free() {
    if (null != layer)
      layer.freeRef();
  }

  public @Override @SuppressWarnings("unused") ActivationLayerTestBase addRef() {
    return (ActivationLayerTestBase) super.addRef();
  }
}
