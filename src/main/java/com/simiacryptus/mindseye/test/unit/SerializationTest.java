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

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.function.Consumer;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

public class SerializationTest extends ComponentTestBase<ToleranceStatistics> {
  @Nonnull
  private final RefHashMap<SerialPrecision, Layer> models = new RefHashMap<>();
  private boolean persist = false;

  @Nonnull
  public RefHashMap<SerialPrecision, Layer> getModels() {
    return RefUtil.addRef(models);
  }

  public boolean isPersist() {
    return persist;
  }

  public void setPersist(boolean persist) {
    this.persist = persist;
  }

  @Nonnull
  public static byte[] compressGZ(@Nonnull String prettyPrint) {
    return compressGZ(prettyPrint.getBytes(Charset.forName("UTF-8")));
  }

  @Nonnull
  public static byte[] compressGZ(byte[] bytes) {
    @Nonnull
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    try {
      try (@Nonnull
           GZIPOutputStream out = new GZIPOutputStream(byteArrayOutputStream)) {
        IOUtils.write(bytes, out);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return byteArrayOutputStream.toByteArray();
  }

  @Nullable
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer,
                                  @Nullable final Tensor... inputPrototype) {
    if (null != inputPrototype)
      RefUtil.freeRef(inputPrototype);
    log.h1("Serialization");
    log.p("This apply will demonstrate the key's JSON serialization, and verify deserialization integrity.");

    String prettyPrint = "";
    log.h2("Raw Json");
    try {
      prettyPrint = log.eval(RefUtil.wrapInterface(() -> {
        final JsonObject json = layer.getJson().getAsJsonObject();
        @Nonnull final Layer echo = Layer.fromJson(json);
        if (layer == echo) {
          echo.freeRef();
          throw new AssertionError("Serialization did not copy");
        }
        if (!layer.equals(echo)) {
          echo.freeRef();
          throw new AssertionError("Serialization not equal");
        }
        echo.freeRef();
        return new GsonBuilder().setPrettyPrinting().create().toJson(json);
      }, layer.addRef()));
      @Nonnull
      String filename = layer.getClass().getSimpleName() + "_" + log.getName() + ".json";
      log.p(log.file(prettyPrint, filename,
          RefString.format("Wrote Model to %s; %s characters", filename, prettyPrint.length())));
    } catch (RuntimeException e) {
      e.printStackTrace();
      Util.sleep(1000);
    } catch (OutOfMemoryError e) {
      e.printStackTrace();
      Util.sleep(1000);
    }
    log.p("");
    @Nonnull
    Object outSync = new Object();
    if (prettyPrint.isEmpty() || prettyPrint.length() > 1024 * 64)
      RefArrays.stream(SerialPrecision.values()).parallel()
          .forEach(RefUtil.wrapInterface((Consumer<? super SerialPrecision>) precision -> {
            try {
              @Nonnull
              File file = new File(log.getResourceDir(), log.getName() + "_" + precision.name() + ".zip");
              layer.writeZip(file, precision);
              @Nonnull final Layer echo = Layer.fromZip(new ZipFile(file));
              RefHashMap<SerialPrecision, Layer> temp_23_0001 = getModels();
              RefUtil.freeRef(temp_23_0001.put(precision, echo.addRef()));
              temp_23_0001.freeRef();
              synchronized (outSync) {
                log.h2(RefString.format("Zipfile %s", precision.name()));
                log.p(log.link(file, RefString.format("Wrote Model apply %s precision to %s; %.3fMiB bytes", precision,
                    file.getName(), file.length() * 1.0 / 0x100000)));
              }
              if (!isPersist())
                file.delete();
              if (layer == echo) {
                echo.freeRef();
                throw new AssertionError("Serialization did not copy");
              }
              if (!layer.equals(echo)) {
                echo.freeRef();
                throw new AssertionError("Serialization not equal");
              }
              echo.freeRef();
            } catch (RuntimeException e) {
              e.printStackTrace();
            } catch (OutOfMemoryError e) {
              e.printStackTrace();
            } catch (ZipException e) {
              e.printStackTrace();
            } catch (IOException e) {
              e.printStackTrace();
            }
          }, layer.addRef()));

    layer.freeRef();
    return null;
  }

  @Nonnull
  @Override
  public String toString() {
    return "SerializationTest{" + "models=" + models + ", persist=" + persist + '}';
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
    models.freeRef();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SerializationTest addRef() {
    return (SerializationTest) super.addRef();
  }
}
