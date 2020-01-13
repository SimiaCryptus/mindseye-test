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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.data.DoubleStatistics;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import javax.annotation.Nonnull;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class PCAUtil {
  @Nonnull
  public static RealMatrix getCovariance(@Nonnull final Supplier<RefStream<double[]>> stream) {
    final int dimension = RefUtil.get(stream.get().findAny()).length;
    final RefList<DoubleStatistics> statList = RefIntStream.range(0, dimension * dimension)
        .mapToObj(i -> new DoubleStatistics()).collect(RefCollectors.toList());
    stream.get().forEach(RefUtil.wrapInterface((Consumer<? super double[]>) array -> {
      for (int i = 0; i < dimension; i++) {
        for (int j = 0; j <= i; j++) {
          statList.get(i * dimension + j).accept(array[i] * array[j]);
        }
      }
      RecycleBin.DOUBLES.recycle(array, array.length);
    }, statList == null ? null : statList.addRef()));
    @Nonnull
    final RealMatrix covariance = new BlockRealMatrix(dimension, dimension);
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j <= i; j++) {
        final double v = statList.get(i + dimension * j).getAverage();
        covariance.setEntry(i, j, v);
        covariance.setEntry(j, i, v);
      }
    }
    if (null != statList)
      statList.freeRef();
    return covariance;
  }

  public static Tensor[] pcaFeatures(final RealMatrix covariance, final int components, final int[] featureDimensions,
      final double power) {
    @Nonnull
    final EigenDecomposition decomposition = new EigenDecomposition(covariance);
    final int[] orderedVectors = RefIntStream.range(0, components).mapToObj(x -> x)
        .sorted(RefComparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
    return RefIntStream.range(0, orderedVectors.length).mapToObj(i -> {
      Tensor temp_19_0002 = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), featureDimensions);
      @Nonnull
      final Tensor src = temp_19_0002.copy();
      if (null != temp_19_0002)
        temp_19_0002.freeRef();
      Tensor temp_19_0003 = src.scale(1.0 / src.rms());
      Tensor temp_19_0001 = temp_19_0003.scale((Math.pow(
          decomposition.getRealEigenvalue(orderedVectors[i]) / decomposition.getRealEigenvalue(orderedVectors[0]),
          power)));
      if (null != temp_19_0003)
        temp_19_0003.freeRef();
      src.freeRef();
      return temp_19_0001;
    }).toArray(i -> new Tensor[i]);
  }

  public static void populatePCAKernel_1(final Tensor kernel, final Tensor[] featureSpaceVectors) {
    final int outputBands = featureSpaceVectors.length;
    @Nonnull
    final int[] filterDimensions = kernel.getDimensions();
    RefUtil.freeRef(kernel.setByCoord(RefUtil.wrapInterface(c -> {
      final int kband = c.getCoords()[2];
      final int outband = kband % outputBands;
      final int inband = (kband - outband) / outputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      final double v = featureSpaceVectors[outband].get(x, y, inband);
      return Double.isFinite(v) ? v : kernel.get(c);
    }, kernel == null ? null : kernel.addRef(), Tensor.addRefs(featureSpaceVectors))));
    if (null != featureSpaceVectors)
      ReferenceCounting.freeRefs(featureSpaceVectors);
    if (null != kernel)
      kernel.freeRef();
  }

  public static void populatePCAKernel_2(final Tensor kernel, final Tensor[] featureSpaceVectors) {
    final int outputBands = featureSpaceVectors.length;
    @Nonnull
    final int[] filterDimensions = kernel.getDimensions();
    RefUtil.freeRef(kernel.setByCoord(RefUtil.wrapInterface(c -> {
      final int kband = c.getCoords()[2];
      final int outband = kband % outputBands;
      final int inband = (kband - outband) / outputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      final double v = featureSpaceVectors[inband].get(x, y, outband);
      return Double.isFinite(v) ? v : kernel.get(c);
    }, kernel == null ? null : kernel.addRef(), Tensor.addRefs(featureSpaceVectors))));
    if (null != featureSpaceVectors)
      ReferenceCounting.freeRefs(featureSpaceVectors);
    if (null != kernel)
      kernel.freeRef();
  }
}
