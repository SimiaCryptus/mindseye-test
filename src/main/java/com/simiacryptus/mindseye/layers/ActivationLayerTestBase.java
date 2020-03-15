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
import com.simiacryptus.mindseye.test.LayerTestBase;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.MustCall;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.junit.After;
import smile.plot.swing.PlotCanvas;
import smile.plot.swing.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Random;
import java.util.function.DoubleFunction;
import java.util.function.Function;

public abstract class ActivationLayerTestBase extends LayerTestBase {

  @Nullable
  @RefIgnore
  private final Layer layer;

  public ActivationLayerTestBase(@Nullable final Layer layer) {
    this.layer = layer;
  }

  @Nullable
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    TrainingTester trainingTester = new TrainingTester() {

      public @SuppressWarnings("unused")
      void _free() {
        super._free();
      }

      @Nonnull
      @Override
      protected Layer lossLayer() {
        return ActivationLayerTestBase.this.lossLayer();
      }
    };
    trainingTester.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    return trainingTester;
  }

  @Nonnull
  public static PlotCanvas plot(final String title, final double[][] data) {
    @Nonnull final PlotCanvas plot = ScatterPlot.plot(data);
    plot.setTitle(title);
    plot.setAxisLabels("x", "y");
    plot.setSize(600, 400);
    return plot;
  }

  @Nonnull
  public static PlotCanvas plot(final String title, @Nonnull final RefList<double[]> plotData,
                                @Nonnull final Function<double[], double[]> function) {
    final double[][] data = plotData.stream().map(function).toArray(double[][]::new);
    plotData.freeRef();
    return ActivationLayerTestBase.plot(title, data);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{2, 3, 1}};
  }

  @Nullable
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return layer == null ? null : layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{{100, 100, 1}};
  }

  @Nonnull
  public RefDoubleStream scan() {
    return RefIntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0);
  }

  @Override
  public void run(@Nonnull final NotebookOutput log) {
    super.run(log);

    log.h3("Function Plots");
    final Layer layer = getLayer(new int[][]{{1}}, new Random());
    final RefList<double[]> plotData = scan().mapToObj(RefUtil.wrapInterface((DoubleFunction<? extends double[]>) x -> {
      @Nonnull
      Tensor tensor = new Tensor(x);
      @Nonnull final SimpleEval eval = SimpleEval.run(layer == null ? null : layer.addRef(), tensor);
      Tensor temp_03_0005 = eval.getOutput();
      Tensor[] derivative = eval.getDerivative();
      assert derivative != null;
      assert temp_03_0005 != null;
      double[] temp_03_0002 = new double[]{x, temp_03_0005.get(0), derivative[0].get(0)};
      RefUtil.freeRef(derivative);
      temp_03_0005.freeRef();
      eval.freeRef();
      return temp_03_0002;
    }, layer)).collect(RefCollectors.toList());

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[]{x[0], x[1]});
    }, plotData == null ? null : plotData.addRef()));

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotCanvas>) () -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[]{x[0], x[2]});
    }, plotData));
  }

  @After
  @MustCall
  public void cleanup() {
    super.cleanup();
    if (null != layer)
      layer.freeRef();
  }

}
