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
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.MustCall;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefDoubleStream;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import smile.plot.swing.Canvas;
import smile.plot.swing.PlotPanel;
import smile.plot.swing.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleFunction;
import java.util.function.Function;

/**
 * The type Activation layer test base.
 */
public abstract class ActivationLayerTestBase extends LayerTestBase {

  @Nullable
  @RefIgnore
  private final Layer layer;

  /**
   * Instantiates a new Activation layer test base.
   *
   * @param layer the layer
   */
  public ActivationLayerTestBase(@Nullable final Layer layer) {
    this.layer = layer;
  }

  @Nonnull
  @Override
  public int[][] getLargeDims() {
    return new int[][]{{100, 100, 1}};
  }

  @Nullable
  @Override
  public Layer getLayer() {
    return layer == null ? null : layer.addRef();
  }

  @Nonnull
  @Override
  public int[][] getSmallDims() {
    return new int[][]{{2, 3, 1}};
  }

  @Nullable
  @Override
  public TrainingTester getTrainingTester() {
    TrainingTester trainingTester = super.getTrainingTester();
    trainingTester.setRandomizationMode(TrainingTester.RandomizationMode.Random);
    return trainingTester;
  }

  /**
   * Plot plot canvas.
   *
   * @param title the title
   * @param data  the data
   * @return the plot canvas
   */
  public static PlotPanel plot(final String title, final double[][] data) {
    @Nonnull final ScatterPlot scatterPlot = ScatterPlot.of(data);
    Canvas canvas = new Canvas(new double[]{0, 0}, new double[]{1, 1e-5}, true);
    PlotPanel plotPanel = new PlotPanel(canvas);
    canvas.add(scatterPlot);
    canvas.setTitle(title);
    canvas.setAxisLabels("x", "y");
    return plotPanel;
  }

  /**
   * Plot plot canvas.
   *
   * @param title    the title
   * @param plotData the plot data
   * @param function the function
   * @return the plot canvas
   */
  public static PlotPanel plot(final String title, @Nonnull final RefList<double[]> plotData,
                               @Nonnull final Function<double[], double[]> function) {
    final double[][] data = plotData.stream().map(function).toArray(double[][]::new);
    plotData.freeRef();
    return ActivationLayerTestBase.plot(title, data);
  }

  /**
   * Scan ref double stream.
   *
   * @return the ref double stream
   */
  @Nonnull
  public RefDoubleStream scan() {
    return RefIntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0);
  }

  /**
   * Activation plot.
   */
  @Test
  @Timeout(value = 15, unit = TimeUnit.MINUTES)
  public void activationPlot() {
    NotebookOutput log = getLog();
    log.h3("Function Plots");
    final Layer layer1 = getLayer();
    final RefList<double[]> plotData = scan().mapToObj(RefUtil.wrapInterface((DoubleFunction<? extends double[]>) x -> {
      @Nonnull
      Tensor tensor = new Tensor(x);
      @Nonnull final SimpleEval eval = SimpleEval.run(layer1 == null ? null : layer1.addRef(), tensor);
      Tensor output = eval.getOutput();
      Tensor[] derivative = eval.getDerivative();
      assert derivative != null;
      assert output != null;
      double[] temp_03_0002 = new double[]{x, output.get(0), derivative[0].get(0)};
      RefUtil.freeRef(derivative);
      output.freeRef();
      eval.freeRef();
      return temp_03_0002;
    }, layer1)).collect(RefCollectors.toList());

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotPanel>) () -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[]{x[0], x[1]});
    }, plotData == null ? null : plotData.addRef()));

    log.eval(RefUtil.wrapInterface((UncheckedSupplier<PlotPanel>) () -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData == null ? null : plotData.addRef(),
          x -> new double[]{x[0], x[2]});
    }, plotData));
  }

  /**
   * Cleanup.
   */
  @AfterEach
  @MustCall
  public void cleanup() {
    if (null != layer)
      layer.freeRef();
  }

}
