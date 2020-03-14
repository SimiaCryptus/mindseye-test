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

import smile.plot.swing.LinePlot;
import smile.plot.swing.Plot;
import smile.plot.swing.ScatterPlot;

import java.awt.*;
import java.util.List;

public class ProblemRun {

  public final Color color;
  public final List<StepRecord> history;
  public final String name;
  public final PlotType type;

  public ProblemRun(final String name, final List<StepRecord> history, final Color color, final PlotType type) {
    this.history = history;
    this.name = name;
    this.color = color;
    this.type = type;
  }

  public Plot plot(final double[][] pts) {
    Plot plot;
    switch (type) {
      case Scatter:
        plot = new ScatterPlot(pts);
        plot.setID(name);
        plot.setColor(color);
        return plot;
      case Line:
        plot = new LinePlot(pts);
        plot.setID(name);
        plot.setColor(color);
        return plot;
      default:
        throw new IllegalStateException(type.toString());
    }
  }

  public enum PlotType {
    Line, Scatter
  }
}
