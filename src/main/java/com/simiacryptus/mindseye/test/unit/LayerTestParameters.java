/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test.unit;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefCollection;
import com.simiacryptus.ref.wrappers.RefHashSet;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * The type Layer test parameters.
 */
class LayerTestParameters extends ReferenceCountingBase {
  @Nullable
  private final Layer layer;
  private final int[][] inputDims;

  /**
   * Instantiates a new Layer test parameters.
   *
   * @param layer     the layer
   * @param inputDims the input dims
   */
  protected LayerTestParameters(@Nullable Layer layer, int[][] inputDims) {
    this.layer = layer;
    this.inputDims = inputDims;
  }

  /**
   * Get dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  public int[][] getDims() {
    return inputDims;
  }

  /**
   * Gets layer.
   *
   * @return the layer
   */
  @Nullable
  public Layer getLayer() {
    return layer == null ? null : layer.addRef();
  }

  /**
   * Gets node tests.
   *
   * @param layer     the layer
   * @param inputDims the input dims
   * @return the node tests
   */
  @Nonnull
  public static RefCollection<LayerTestParameters> getNodeTests(@Nonnull Layer layer, @Nonnull int[][] inputDims) {
    @Nonnull
    DAGNetwork layerCopy = (DAGNetwork) layer.copy();
    layer.freeRef();
    @Nonnull
    RefHashSet<LayerTestParameters> layerTestParameters = new RefHashSet<>();
    layerCopy.visitNodes(RefUtil.wrapInterface(node -> {
      @Nullable
      Layer inner = node.getLayer();
      @Nullable
      Layer wrapper = new LayerBase() {
        {
          inner.addRef();
          layerTestParameters.addRef();
        }

        @Nullable
        @Override
        public Result eval(@Nonnull Result... array) {
          if (null == inner) {
            RefUtil.freeRef(array);
            return null;
          }
          @Nullable
          Result result = inner.eval(RefUtil.addRef(array));
          layerTestParameters.add(
              new LayerTestParameters(inner.addRef(), RefArrays.stream(array).map(x -> {
                return LayerTests.getDimensions(LayerTests.getData(x));
              }).toArray(int[][]::new)));
          return result;
        }

        @Override
        public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
          assert inner != null;
          return inner.getJson(resources, dataSerializer).getAsJsonObject();
        }

        @Nullable
        @Override
        public RefList<double[]> state() {
          assert inner != null;
          return inner.state();
        }

        public void _free() {
          super._free();
          inner.freeRef();
          layerTestParameters.freeRef();
        }
      };
      if (null != inner)
        inner.freeRef();
      node.setLayer(wrapper);
      node.freeRef();
    }, layerTestParameters.addRef()));
    Tensor[] input = RefArrays.stream(inputDims).map(Tensor::new).toArray(Tensor[]::new);
    Result eval = layerCopy.eval(input);
    layerCopy.freeRef();
    assert eval != null;
    eval.freeRef();
    return layerTestParameters;
  }

  @Override
  @RefIgnore
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    LayerTestParameters that = (LayerTestParameters) o;
    return Objects.equals(layer, that.layer) &&
        Arrays.equals(inputDims, that.inputDims);
  }

  @Override
  @RefIgnore
  public int hashCode() {
    int result = Objects.hash(layer);
    result = 31 * result + Arrays.hashCode(inputDims);
    return result;
  }

  public void _free() {
    if (null != layer)
      layer.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LayerTestParameters addRef() {
    return (LayerTestParameters) super.addRef();
  }
}
