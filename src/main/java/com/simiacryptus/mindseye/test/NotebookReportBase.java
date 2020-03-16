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

import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.simiacryptus.aws.S3Util;
import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.wrappers.RefConsumer;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.CodeUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.TestInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.FileNotFoundException;
import java.net.URI;
import java.text.SimpleDateFormat;
import java.util.Date;

public abstract class NotebookReportBase {

  protected static final Logger logger = LoggerFactory.getLogger(NotebookReportBase.class);

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  @Nonnull
  protected String reportingFolder = "reports/_reports";

  public Class<? extends NotebookReportBase> getReportClass() {
    return getClass();
  }

  @Nonnull
  public abstract ReportType getReportType();

  protected abstract Class<?> getTargetClass();

  @Nullable
  public static CharSequence setReportType(@Nonnull NotebookOutput log, @Nullable Class<?> networkClass,
                                           final CharSequence prefix) {
    if (null == networkClass)
      return null;
    @Nullable
    String javadoc = CodeUtil.getJavadoc(networkClass);
    log.setMetadata(prefix + "_class_short", networkClass.getSimpleName());
    log.setMetadata(prefix + "_class_full", networkClass.getCanonicalName());
    assert javadoc != null;
    log.setMetadata(prefix + "_class_doc", javadoc.replaceAll("\n", ""));
    return javadoc;
  }

  @Nonnull
  public static File getTestReportLocation(TestInfo testInfo, @Nonnull final Class<?> sourceClass) {
    @Nonnull
    File path = new File(Util.mkString(File.separator,
        TestSettings.INSTANCE.testRepo,
        toPathString(sourceClass),
        testInfo.getTestClass().map(c -> c.getSimpleName()).orElse(""),
        testInfo.getTestMethod().get().getName(),
        new SimpleDateFormat("yyyyMMddmmss").format(new Date())
    ));
    path.getParentFile().mkdirs();
    logger.info(RefString.format("Output Location: %s", path.getAbsoluteFile()));
    return path;
  }

  @NotNull
  public static String toPathString(@Nonnull Class<?> sourceClass) {
    return toPathString(sourceClass, File.separatorChar);
  }

  @NotNull
  public static String toPathString(@Nonnull Class<?> sourceClass, char separatorChar) {
    return sourceClass.getCanonicalName()
        .replace('.', separatorChar)
        .replace('$', separatorChar);
  }

  @Nonnull
  public NotebookOutput getLog(TestInfo testInfo) {
    Class<?> targetClass = getTargetClass();
    final File reportFile = getTestReportLocation(testInfo, targetClass);
    try {
      MarkdownNotebookOutput markdownNotebookOutput = new MarkdownNotebookOutput(
          reportFile, true, testInfo.getTestMethod().get().getName()
      );
      markdownNotebookOutput.setEnableZip(false);
      URI testArchive = TestSettings.INSTANCE.testArchive;
      if (null != testArchive) markdownNotebookOutput.setArchiveHome(testArchive.resolve(
          Util.mkString("/",
              toPathString(targetClass, '/'),
              testInfo.getTestClass().map(c -> c.getSimpleName()).orElse(""),
              testInfo.getTestMethod().get().getName(),
              new SimpleDateFormat("yyyyMMddmmss").format(new Date())
          )
      ));
      S3Util.uploadOnComplete(markdownNotebookOutput, AmazonS3ClientBuilder.standard().build());
      File metadataLocation = new File(TestSettings.INSTANCE.testRepo, "registry");
      metadataLocation.mkdirs();
      markdownNotebookOutput.setMetadataLocation(metadataLocation);
      return markdownNotebookOutput;
    } catch (FileNotFoundException e) {
      throw Util.throwException(e);
    }
  }

  public void printHeader(@Nonnull NotebookOutput log) {
    log.setMetadata("created_on", new Date().toString());
    log.setMetadata("report_type", getReportType().name());
    CharSequence targetDescription = setReportType(log, getTargetClass(), "network");
    if (null != targetDescription && targetDescription.length() > 0) {
      log.p("__Target Description:__ " + targetDescription);
    }
    CharSequence reportDescription = setReportType(log, getReportClass(), "report");
    if (null != reportDescription && reportDescription.length() > 0) {
      log.p("__Report Description:__ " + reportDescription);
    }
  }

  public void report(TestInfo testInfo, @Nonnull RefConsumer<NotebookOutput> fn) {
    try (@Nonnull NotebookOutput log = getLog(testInfo)) {
      CodeUtil.withRefLeakMonitor(log, withRef -> {
        printHeader(withRef);
        @Nonnull
        TimedResult<Void> time = TimedResult.time(() -> {
          try {
            fn.accept(withRef);
            withRef.setMetadata("result", "OK");
          } catch (Throwable e) {
            withRef.setMetadata("result", MarkdownNotebookOutput.replaceAll(MarkdownNotebookOutput.getExceptionString(e).toString(), "\n", "<br/>").trim());
            throw Util.throwException(e);
          }
        });
        withRef.setMetadata("execution_time", RefString.format("%.6f", time.timeNanos / 1e9));
        withRef.setMetadata("gc_time", RefString.format("%.6f", time.gcMs / 1e9));
        time.freeRef();
      });
    } catch (Throwable e) {
      throw Util.throwException(e);
    }
  }

  public enum ReportType {
    Applications, Components, Models, Data, Optimizers, Experiments
  }

}
