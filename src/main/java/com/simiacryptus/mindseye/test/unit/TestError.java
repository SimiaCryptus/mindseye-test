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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import java.util.Arrays;

public @RefAware
class TestError extends RuntimeException implements ReferenceCounting {
  public final ComponentTest<?> test;
  @Nonnull
  public final Layer layer;
  @RefIgnore
  private final ReferenceCountingBase refCounter = new ReferenceCountingBase() {
    public void _free() {
      TestError.this._free();
    }
  };

  public TestError(Throwable cause, ComponentTest<?> test, @Nonnull Layer layer) {
    super(String.format("Error in %s apply %s", test, layer), cause);
    this.test = test;
    this.layer = layer;
    layer.detach();
  }

  @RefIgnore
  @Override
  public boolean isFinalized() {
    return refCounter.isFinalized();
  }

  public static @SuppressWarnings("unused")
  TestError[] addRefs(TestError[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TestError::addRef)
        .toArray((x) -> new TestError[x]);
  }

  public static @SuppressWarnings("unused")
  TestError[][] addRefs(TestError[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TestError::addRefs)
        .toArray((x) -> new TestError[x][]);
  }

  public void _free() {
    layer.freeRef();
    test.freeRef();
  }

  @RefIgnore
  @Override
  public boolean assertAlive() {
    return refCounter.assertAlive();
  }

  @RefIgnore
  @Override
  public int currentRefCount() {
    return refCounter.currentRefCount();
  }

  @RefIgnore
  @Override
  public @NotNull ReferenceCounting detach() {
    return refCounter.detach();
  }

  @RefIgnore
  @Override
  public void freeRefAsync() {
    refCounter.freeRefAsync();
  }

  @RefIgnore
  @Override
  public boolean tryAddRef() {
    return refCounter.tryAddRef();
  }

  public @Override
  @SuppressWarnings("unused")
  TestError addRef() {
    refCounter.addRef();
    return this;
  }
}
