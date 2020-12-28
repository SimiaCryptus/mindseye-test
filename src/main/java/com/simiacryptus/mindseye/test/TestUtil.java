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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.FileHTTPD;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.io.GifSequenceWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.swing.Canvas;
import smile.plot.swing.PlotPanel;
import smile.plot.swing.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

/**
 * The type Test util.
 */
public class TestUtil {
  /**
   * The constant S3_ROOT.
   */
  public static final URI S3_ROOT = URI.create("https://s3-us-west-2.amazonaws.com/simiacryptus/");
  private static final Logger logger = LoggerFactory.getLogger(TestUtil.class);

  /**
   * Gets stack info.
   *
   * @return the stack info
   */
  public static Map<String, List<String>> getStackInfo() {
    return Thread.getAllStackTraces().entrySet().stream().collect(Collectors.toMap(entry -> {
      Thread key = entry.getKey();
      RefUtil.freeRef(entry);
      return RefString.format("%s@%d", key.getName(), key.getId());
    }, entry -> {
      StackTraceElement[] value = entry.getValue();
      RefUtil.freeRef(entry);
      return Arrays.stream(value).map(StackTraceElement::toString).collect(Collectors.toList());
    }));
  }

  /**
   * Compare plot canvas.
   *
   * @param title  the title
   * @param trials the trials
   * @return the plot canvas
   */
  @Nullable
  public static PlotPanel compare(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics xStatistics = RefArrays.stream(trials)
          .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.iteration)).filter(Double::isFinite)
          .summaryStatistics();
      final DoubleSummaryStatistics yStatistics = RefArrays.stream(trials)
          .flatMapToDouble(
              x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
          .filter(Double::isFinite).summaryStatistics();
      if (xStatistics.getCount() == 0) {
        logger.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {xStatistics.getMin(),
          yStatistics.getCount() < 2 ? 0 : yStatistics.getMin()};
      @Nonnull final double[] upperBound = {xStatistics.getMax(),
          yStatistics.getCount() < 2 ? 1 : yStatistics.getMax()};
      Canvas canvas = new Canvas(new double[]{0, 0}, new double[]{1, 1e-5}, true);
      PlotPanel plotPanel = new PlotPanel(canvas);
      canvas.setTitle(title);
      canvas.setAxisLabels("x", "y");
      plotPanel.setSize(600, 400);
      canvas.setAxisLabels("Iteration", "log10(Fitness)");
      final RefList<ProblemRun> filtered = RefArrays.stream(trials).filter(x -> !x.history.isEmpty())
          .collect(RefCollectors.toList());
      if (filtered.isEmpty()) {
        logger.info("No Data");
        filtered.freeRef();
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream())
          .mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      logger.info(RefString.format("Plotting range=%s, %s; valueStats=%s", RefArrays.toString(lowerBound),
          RefArrays.toString(upperBound), valueStatistics));
      filtered.forEach(trial -> {
        final double[][] pts = trial.history.stream()
            .map(step -> new double[]{step.iteration, Math.log10(Math.max(step.fitness, valueStatistics.getMin()))})
            .filter(x -> RefArrays.stream(x).allMatch(Double::isFinite)).toArray(double[][]::new);
        if (pts.length > 1) {
          logger.info(RefString.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        } else {
          logger.info(RefString.format("Only %s points for %s", pts.length, trial.name));
        }
      });
      filtered.freeRef();
      return plotPanel;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }

  /**
   * Compare time plot canvas.
   *
   * @param title  the title
   * @param trials the trials
   * @return the plot canvas
   */
  @Nullable
  public static PlotPanel compareTime(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics[] xStatistics = RefArrays.stream(trials)
          .map(x -> x.history.stream().mapToDouble(step -> step.epochTime).filter(Double::isFinite).summaryStatistics())
          .toArray(DoubleSummaryStatistics[]::new);
      final double totalTime = RefArrays.stream(xStatistics).mapToDouble(x -> x.getMax() - x.getMin()).max()
          .getAsDouble();
      final DoubleSummaryStatistics yStatistics = RefArrays.stream(trials)
          .flatMapToDouble(
              x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
          .filter(Double::isFinite).summaryStatistics();
      if (yStatistics.getCount() == 0) {
        logger.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {0, yStatistics.getMin()};
      @Nonnull final double[] upperBound = {totalTime / 1000.0, yStatistics.getCount() == 1 ? 0 : yStatistics.getMax()};
      Canvas canvas = new Canvas(lowerBound, upperBound);
      PlotPanel plotPanel = new PlotPanel(canvas);
      canvas.setTitle(title);
      canvas.setAxisLabels("x", "y");
      canvas.setAxisLabels("Time", "log10(Fitness)");
      plotPanel.setSize(600, 400);
      final RefList<ProblemRun> filtered = RefArrays.stream(trials).filter(x -> !x.history.isEmpty())
          .collect(RefCollectors.toList());
      if (filtered.isEmpty()) {
        logger.info("No Data");
        filtered.freeRef();
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream())
          .mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      logger.info(RefString.format("Plotting range=%s, %s; valueStats=%s", RefArrays.toString(lowerBound),
          RefArrays.toString(upperBound), valueStatistics));
      for (int t = 0; t < filtered.size(); t++) {
        final ProblemRun trial = filtered.get(t);
        final DoubleSummaryStatistics trialStats = xStatistics[t];
        final double[][] pts = trial.history.stream().map(step -> {
          return new double[]{(step.epochTime - trialStats.getMin()) / 1000.0,
              Math.log10(Math.max(step.fitness, valueStatistics.getMin()))};
        }).filter(x -> RefArrays.stream(x).allMatch(Double::isFinite)).toArray(double[][]::new);
        if (pts.length > 1) {
          logger.info(RefString.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        } else {
          logger.info(RefString.format("Only %s points for %s", pts.length, trial.name));
        }
      }
      filtered.freeRef();
      return plotPanel;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }

  /**
   * Extract performance.
   *
   * @param log     the log
   * @param network the network
   */
  public static void extractPerformance(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    log.p("Per-key Performance Metrics:");
    log.run(RefUtil.wrapInterface(() -> {
      @Nonnull final RefMap<CharSequence, MonitoringWrapperLayer> metrics = new RefHashMap<>();
      network.visitNodes(RefUtil.wrapInterface(node -> {
        Layer nodeLayer = node.getLayer();
        if (nodeLayer instanceof MonitoringWrapperLayer) {
          @Nullable final MonitoringWrapperLayer layer = (MonitoringWrapperLayer) nodeLayer.addRef();
          Layer inner = layer.getInner();
          assert inner != null;
          String str = inner.toString();
          str += " class=" + inner.getClass().getName();
          inner.freeRef();
          RefUtil.freeRef(metrics.put(str, layer.addRef()));
          layer.freeRef();
        }
        assert nodeLayer != null;
        nodeLayer.freeRef();
        node.freeRef();
      }, metrics.addRef()));
      RefSet<Map.Entry<CharSequence, MonitoringWrapperLayer>> temp_13_0018 = metrics.entrySet();
      TestUtil.logger.info("Performance: \n\t" + RefUtil.orElse(temp_13_0018.stream()
          .sorted(RefComparator.comparingDouble(new ToDoubleFunction<Map.Entry<CharSequence, MonitoringWrapperLayer>>() {
            @Override
            @RefIgnore
            public double applyAsDouble(Map.Entry<CharSequence, MonitoringWrapperLayer> x) {
              MonitoringWrapperLayer temp_13_0019 = x.getValue();
              double temp_13_0002 = -temp_13_0019.getForwardPerformance().getMean();
              temp_13_0019.freeRef();
              RefUtil.freeRef(x);
              return temp_13_0002;
            }
          })).map(e -> {
            MonitoringWrapperLayer temp_13_0020 = e.getValue();
            @Nonnull final PercentileStatistics performanceF = temp_13_0020.getForwardPerformance();
            temp_13_0020.freeRef();
            MonitoringWrapperLayer temp_13_0021 = e.getValue();
            @Nonnull final PercentileStatistics performanceB = temp_13_0021.getBackwardPerformance();
            temp_13_0021.freeRef();
            String temp_13_0003 = RefString.format("%.6fs +- %.6fs (%d) <- %s", performanceF.getMean(),
                performanceF.getStdDev(), performanceF.getCount(), e.getKey())
                + (performanceB.getCount() == 0 ? ""
                : RefString.format("%n\tBack: %.6fs +- %.6fs (%s)", performanceB.getMean(), performanceB.getStdDev(),
                performanceB.getCount()));
            RefUtil.freeRef(e);
            return temp_13_0003;
          }).reduce((a, b) -> a + "\n\t" + b), "-"));
      temp_13_0018.freeRef();
      metrics.freeRef();
    }, network.addRef()));
    removeInstrumentation(network);
  }

  /**
   * Remove instrumentation.
   *
   * @param network the network
   */
  public static void removeInstrumentation(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      Layer nodeLayer = node.getLayer();
      if (nodeLayer instanceof MonitoringWrapperLayer) {
        MonitoringWrapperLayer temp_13_0022 = node.<MonitoringWrapperLayer>getLayer();
        Layer layer = temp_13_0022.getInner();
        temp_13_0022.freeRef();
        node.setLayer(layer == null ? null : layer.addRef());
        if (null != layer)
          layer.freeRef();
      }
      if (null != nodeLayer)
        nodeLayer.freeRef();
      node.freeRef();
    });
    network.freeRef();
  }

  /**
   * Sample performance ref map.
   *
   * @param network the network
   * @return the ref map
   */
  @Nonnull
  public static RefMap<CharSequence, Object> samplePerformance(@Nonnull final DAGNetwork network) {
    @Nonnull final RefMap<CharSequence, Object> metrics = new RefHashMap<>();
    network.visitLayers(RefUtil.wrapInterface(layer -> {
      if (layer instanceof MonitoringWrapperLayer) {
        MonitoringWrapperLayer monitoringWrapperLayer = (MonitoringWrapperLayer) layer;
        Layer inner = monitoringWrapperLayer.getInner();
        assert inner != null;
        String str = inner.toString();
        str += " class=" + inner.getClass().getName();
        inner.freeRef();
        RefHashMap<CharSequence, Object> row = new RefHashMap<>();
        RefUtil.freeRef(row.put("fwd", monitoringWrapperLayer.getForwardPerformance().getMetrics()));
        RefUtil.freeRef(row.put("rev", monitoringWrapperLayer.getBackwardPerformance().getMetrics()));
        monitoringWrapperLayer.freeRef();
        RefUtil.freeRef(metrics.put(str, row));
      } else if (null != layer) layer.freeRef();
    }, metrics.addRef()));
    network.freeRef();
    return metrics;
  }

  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  @Nonnull
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history) {
    return getMonitor(history, null);
  }

  /**
   * Gets monitor.
   *
   * @param history the history
   * @param network the network
   * @return the monitor
   */
  @Nonnull
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history, @Nullable final Layer network) {
    if (null != network)
      network.freeRef();
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(final String msg) {
        logger.info(msg);
        super.log(msg);
      }

      @Override
      public void onStepComplete(@Nonnull final Step currentPoint) {
        assert currentPoint.point != null;
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
        super.onStepComplete(currentPoint);
      }
    };
  }

  /**
   * Instrument performance.
   *
   * @param network the network
   */
  public static void instrumentPerformance(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      Layer layer = node.getLayer();
      if (layer instanceof MonitoringWrapperLayer) {
        ((MonitoringWrapperLayer) layer).shouldRecordSignalMetrics(false);
        RefUtil.freeRef(((MonitoringWrapperLayer) layer).addRef());
      } else {
        MonitoringWrapperLayer temp_13_0015 = new MonitoringWrapperLayer(layer == null ? null : layer.addRef());
        temp_13_0015.shouldRecordSignalMetrics(false);
        @Nonnull
        MonitoringWrapperLayer monitoringWrapperLayer = temp_13_0015.addRef();
        temp_13_0015.freeRef();
        node.setLayer(monitoringWrapperLayer);
      }
      if (null != layer)
        layer.freeRef();
      node.freeRef();
    });
    network.freeRef();
  }

  /**
   * Plot j panel.
   *
   * @param history the history
   * @return the j panel
   */
  @Nullable
  public static JPanel plot(@Nonnull final List<StepRecord> history) {
    try {
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).summaryStatistics();
      double min = valueStats.getMin();
      if (0 < min) {
        double[][] data = history.stream()
            .map(step -> new double[]{step.iteration, Math.log10(Math.max(min, step.fitness))})
            .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(double[][]::new);
        if (Arrays.stream(data).mapToInt(x -> x.length).sum() == 0)
          return null;
        Canvas canvas = new Canvas(new double[]{0, 0}, new double[]{1, 1e-5}, true);
        PlotPanel plotPanel = new PlotPanel(canvas);
        canvas.add(ScatterPlot.of(data));
        canvas.setTitle("Convergence Plot");
        canvas.setAxisLabels("Iteration", "log10(Fitness)");
        plotPanel.setSize(600, 400);
        return plotPanel;
      } else {
        double[][] data = history.stream().map(step -> new double[]{step.iteration, step.fitness})
            .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(double[][]::new);
        if (Arrays.stream(data).mapToInt(x -> x.length).sum() == 0)
          return null;
        Canvas canvas = new Canvas(new double[]{0, 0}, new double[]{1, 1e-5}, true);
        PlotPanel plotPanel = new PlotPanel(canvas);
        plotPanel.setSize(600, 400);
        canvas.add(ScatterPlot.of(data));
        canvas.setTitle("Convergence Plot");
        canvas.setAxisLabels("Iteration", "Fitness");
        return plotPanel;
      }
    } catch (@Nonnull final Exception e) {
      logger.warn("Error plotting", e);
      return null;
    }
  }

  /**
   * Plot time plot canvas.
   *
   * @param history the history
   * @return the plot canvas
   */
  @Nullable
  public static PlotPanel plotTime(@Nonnull final List<StepRecord> history) {
    try {
      final LongSummaryStatistics timeStats = history.stream().mapToLong(x -> x.epochTime).summaryStatistics();
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0)
          .summaryStatistics();
      Canvas canvas = new Canvas(new double[]{0, 0}, new double[]{1, 1e-5}, true);
      PlotPanel plotPanel = new PlotPanel(canvas);
      canvas.add(ScatterPlot.of(history.stream()
              .map(step -> new double[]{(step.epochTime - timeStats.getMin()) / 1000.0,
                      Math.log10(Math.max(valueStats.getMin(), step.fitness))})
              .filter(x -> Arrays.stream(x).allMatch(Double::isFinite)).toArray(double[][]::new)));
      canvas.setTitle("Convergence Plot");
      canvas.setAxisLabels("Time", "log10(Fitness)");
      plotPanel.setSize(600, 400);
      return plotPanel;
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }

  /**
   * Shuffle ref int stream.
   *
   * @param stream the stream
   * @return the ref int stream
   */
  @Nonnull
  public static RefIntStream shuffle(@Nonnull RefIntStream stream) {
    // http://primes.utm.edu/lists/small/10000.txt
    long coprimeA = 41387;
    long coprimeB = 9967;
    long ringSize = coprimeA * coprimeB - 1;
    @Nonnull
    IntToLongFunction fn = x -> x * coprimeA * coprimeA % ringSize;
    @Nonnull
    LongToIntFunction inv = x -> (int) (x * coprimeB * coprimeB % ringSize);
    @Nonnull
    IntUnaryOperator conditions = x -> {
      assert x < ringSize;
      assert x >= 0;
      return x;
    };
    return stream.map(conditions).mapToLong(fn).sorted().mapToInt(inv);
  }

  /**
   * Or else supplier.
   *
   * @param <T>       the type parameter
   * @param suppliers the suppliers
   * @return the supplier
   */
  @Nonnull
  public static <T> Supplier<T> orElse(@Nonnull Supplier<T>... suppliers) {
    return () -> {
      for (@Nonnull
          Supplier<T> supplier : suppliers) {
        T t = supplier.get();
        if (null != t)
          return t;
      }
      return null;
    };
  }

  /**
   * Animated gif char sequence.
   *
   * @param log    the log
   * @param images the images
   * @return the char sequence
   */
  @Nonnull
  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, @Nonnull final BufferedImage... images) {
    return animatedGif(log, 15000, images);
  }

  /**
   * Build map ref map.
   *
   * @param <K>       the type parameter
   * @param <V>       the type parameter
   * @param configure the configure
   * @return the ref map
   */
  @Nonnull
  public static <K, V> RefMap<K, V> buildMap(@Nonnull RefConsumer<RefMap<K, V>> configure) {
    RefMap<K, V> map = new RefHashMap<>();
    configure.accept(map.addRef());
    return map;
  }

  /**
   * Geometric stream supplier.
   *
   * @param start the start
   * @param end   the end
   * @param steps the steps
   * @return the supplier
   */
  @Nonnull
  public static Supplier<RefDoubleStream> geometricStream(final double start, final double end, final int steps) {
    double step = Math.pow(end / start, 1.0 / (steps - 1));
    return () -> RefDoubleStream.iterate(start, x -> x * step).limit(steps);
  }

  /**
   * Shuffle ref list.
   *
   * @param <T>  the type parameter
   * @param list the list
   * @return the ref list
   */
  @Nonnull
  public static <T> RefList<T> shuffle(@Nullable final RefList<T> list) {
    RefArrayList<T> copy = new RefArrayList<>(list);
    RefCollections.shuffle(copy.addRef());
    return copy;
  }

  /**
   * Add global handlers.
   *
   * @param httpd the httpd
   */
  public static void addGlobalHandlers(@Nullable final FileHTTPD httpd) {
    if (null != httpd) {
      httpd.addGET("threads.json", "text/json", out -> {
        try {
          JsonUtil.getMapper().writer().writeValue(out, getStackInfo());
          //JsonUtil.MAPPER.writer().writeValue(out, new HashMap<>());
          out.close();
        } catch (IOException e) {
          throw Util.throwException(e);
        }
      });
    }
  }

  /**
   * Sum tensor.
   *
   * @param tensorStream the tensor stream
   * @return the tensor
   */
  @Nonnull
  public static Tensor sum(@Nonnull RefCollection<Tensor> tensorStream) {
    Tensor temp_13_0012 = RefUtil.get(tensorStream.stream().reduce((a, b) -> {
      return Tensor.add(a, b);
    }));
    tensorStream.freeRef();
    return temp_13_0012;
  }

  /**
   * Sum tensor.
   *
   * @param tensorStream the tensor stream
   * @return the tensor
   */
  @Nonnull
  public static Tensor sum(@Nonnull RefStream<Tensor> tensorStream) {
    return RefUtil.get(tensorStream.reduce((a, b) -> {
      return Tensor.add(a, b);
    }));
  }

  /**
   * Avg tensor.
   *
   * @param values the values
   * @return the tensor
   */
  @Nonnull
  public static Tensor avg(@Nonnull RefCollection<? extends Tensor> values) {
    Tensor temp_13_0027 = sum(values.stream().map(x -> {
      return x;
    }));
    temp_13_0027.scaleInPlace(1.0 / values.size());
    Tensor temp_13_0013 = temp_13_0027.addRef();
    temp_13_0027.freeRef();
    values.freeRef();
    return temp_13_0013;
  }

  /**
   * Render char sequence.
   *
   * @param log       the log
   * @param tensor    the tensor
   * @param normalize the normalize
   * @return the char sequence
   */
  @Nonnull
  public static CharSequence render(@Nonnull final NotebookOutput log, @Nonnull final Tensor tensor,
                                    final boolean normalize) {
    return RefUtil.get(ImageUtil.renderToImages(tensor, normalize).map(image -> {
      return log.png(image, "");
    }).reduce((a, b) -> a + b));
  }

  /**
   * Animated gif char sequence.
   *
   * @param log        the log
   * @param loopTimeMs the loop time ms
   * @param images     the images
   * @return the char sequence
   */
  @Nonnull
  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, final int loopTimeMs,
                                         @Nonnull final BufferedImage... images) {
    try {
      @Nonnull
      String filename = UUID.randomUUID().toString() + ".gif";
      @Nonnull
      File file = new File(log.getResourceDir(), filename);
      GifSequenceWriter.write(file, loopTimeMs / images.length, true, images);
      return RefString.format("<img src=\"etc/%s\" />", filename);
    } catch (IOException e) {
      throw Util.throwException(e);
    }
  }

}
