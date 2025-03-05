/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.th.ThaiAnalyzer;
import org.apache.lucene.benchmark.byTask.feeds.DemoHTMLParser.Parser;
import org.apache.lucene.demo.knn.DemoEmbeddings;
import org.apache.lucene.demo.knn.KnnVectorDict;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KeywordField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;
import org.xml.sax.SAXException;

/**
 * Index all text files under a directory.
 *
 * <p>This is a command-line application demonstrating simple Lucene indexing. Run it with no
 * command-line arguments for usage information.
 */
public class IndexFilesWithMoreInfo implements AutoCloseable {
  static final String KNN_DICT = "knn-dict";
  private static Path docDir;
  // Calculates embedding vectors for KnnVector search
  private final DemoEmbeddings demoEmbeddings;
  private final KnnVectorDict vectorDict;

  private IndexFilesWithMoreInfo(KnnVectorDict vectorDict) throws IOException {
    if (vectorDict != null) {
      this.vectorDict = vectorDict;
      demoEmbeddings = new DemoEmbeddings(vectorDict);
    } else {
      this.vectorDict = null;
      demoEmbeddings = null;
    }
  }

  /** Index all text files under a directory. */
  public static void main(String[] args) throws Exception {
    String usage =
        "java org.apache.lucene.demo.IndexFiles"
            + " [-index INDEX_PATH] [-docs DOCS_PATH] [-update] [-knn_dict DICT_PATH]\n\n"
            + "This indexes the documents in DOCS_PATH, creating a Lucene index "
            + "in INDEX_PATH that can be searched with SearchFiles\n"
            + "IF DICT_PATH contains a KnnVector dictionary, the index will also support KnnVector search";
    String indexPath = "index";
    String docsPath = null;
    String vectorDictSource = null;
    boolean create = true;
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "-index":
          indexPath = args[++i];
          break;
        case "-docs":
          docsPath = args[++i];
          break;
        case "-knn_dict":
          vectorDictSource = args[++i];
          break;
        case "-update":
          create = false;
          break;
        case "-create":
          create = true;
          break;
        default:
          throw new IllegalArgumentException("unknown parameter " + args[i]);
      }
    }

    if (docsPath == null) {
      System.err.println("Usage: " + usage);
      System.exit(1);
    }

    docDir = Paths.get(docsPath);
    if (!Files.isReadable(docDir)) {
      System.out.println(
          "Document directory '"
              + docDir.toAbsolutePath()
              + "' does not exist or is not readable, please check the path");
      System.exit(1);
    }

    Date start = new Date();
    try {
      System.out.println("Indexing to directory '" + indexPath + "'...");

      Directory dir = FSDirectory.open(Paths.get(indexPath));
      Analyzer analyzer = new SimpleAnalyzer();
      IndexWriterConfig iwc = new IndexWriterConfig(analyzer);

      if (create) {
        // Create a new index in the directory, removing any
        // previously indexed documents:
        iwc.setOpenMode(OpenMode.CREATE);
      } else {
        // Add new documents to an existing index:
        iwc.setOpenMode(OpenMode.CREATE_OR_APPEND);
      }

      // Optional: for better indexing performance, if you
      // are indexing many documents, increase the RAM
      // buffer.  But if you do this, increase the max heap
      // size to the JVM (e.g. add -Xmx512m or -Xmx1g):
      //
      // iwc.setRAMBufferSizeMB(256.0);

      KnnVectorDict vectorDictInstance = null;
      long vectorDictSize = 0;
      if (vectorDictSource != null) {
        KnnVectorDict.build(Paths.get(vectorDictSource), dir, KNN_DICT);
        vectorDictInstance = new KnnVectorDict(dir, KNN_DICT);
        vectorDictSize = vectorDictInstance.ramBytesUsed();
      }

      try (IndexWriter writer = new IndexWriter(dir, iwc);
          IndexFilesWithMoreInfo indexFiles = new IndexFilesWithMoreInfo(vectorDictInstance)) {
        indexFiles.indexDocs(writer, docDir);

        // NOTE: if you want to maximize search performance,
        // you can optionally call forceMerge here.  This can be
        // a terribly costly operation, so generally it's only
        // worth it when your index is relatively static (ie
        // you're done adding documents to it):
        //
        // writer.forceMerge(1);
      }  catch (ArrayIndexOutOfBoundsException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
      } finally {
        IOUtils.close(vectorDictInstance);
      }

      Date end = new Date();
      try (IndexReader reader = DirectoryReader.open(dir)) {
        System.out.println(
            "Indexed "
                + reader.numDocs()
                + " documents in "
                + (end.getTime() - start.getTime())
                + " ms");
        if (Objects.isNull(vectorDictSource) == false
            && reader.numDocs() > 100
            && vectorDictSize < 1_000_000
            && System.getProperty("smoketester") == null) {
          throw new RuntimeException(
              "Are you (ab)using the toy vector dictionary? See the package javadocs to understand why you got this exception.");
        }
      }
    } catch (IOException e) {
      System.out.println(" caught a " + e.getClass() + "\n with message: " + e.getMessage());
    }
  }

  /**
   * Indexes the given file using the given writer, or if a directory is given, recurses over files
   * and directories found under the given directory.
   *
   * <p>NOTE: This method indexes one document per input file. This is slow. For good throughput,
   * put multiple documents into your input file(s). An example of this is in the benchmark module,
   * which can create "line doc" files, one document per line, using the <a
   * href="../../../../../contrib-benchmark/org/apache/lucene/benchmark/byTask/tasks/WriteLineDocTask.html"
   * >WriteLineDocTask</a>.
   *
   * @param writer Writer to the index where the given file/dir info will be stored
   * @param path The file to index, or the directory to recurse into to find files to index
   * @throws IOException If there is a low-level I/O error
   */
  void indexDocs(final IndexWriter writer, Path path) throws IOException {
    if (Files.isDirectory(path)) {
      try {
    	  Files.walkFileTree(
          path,
          new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
              try {
                indexDoc(writer, file, attrs.lastModifiedTime().toMillis());
              } catch (
                  @SuppressWarnings("unused")
                  IOException ignore) {
                ignore.printStackTrace(System.err);
                // don't index files that can't be read.
              }
              return FileVisitResult.CONTINUE;
            }
          });
      } catch (ArrayIndexOutOfBoundsException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
      }
    } else {
      indexDoc(writer, path, Files.getLastModifiedTime(path).toMillis());
    }
  }
  
  private static List<String> japanPrefectures = Arrays.asList(
          "Hokkaido", "Aomori", "Iwate", "Miyagi", "Akita", "Yamagata", "Fukushima",
          "Ibaraki", "Tochigi", "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa",
          "Niigata", "Toyama", "Ishikawa", "Fukui", "Yamanashi", "Nagano", "Gifu",
          "Shizuoka", "Aichi", "Mie", "Shiga", "Kyoto", "Osaka", "Hyogo", "Nara",
          "Wakayama", "Tottori", "Shimane", "Okayama", "Hiroshima", "Yamaguchi",
          "Tokushima", "Kagawa", "Ehime", "Kochi", "Fukuoka", "Saga", "Nagasaki",
          "Kumamoto", "Oita", "Miyazaki", "Kagoshima", "Okinawa"
      );
  
  void indexDoc(IndexWriter writer, Path file, long lastModified) throws IOException {
	    try (InputStream stream = Files.newInputStream(file)) {
	    	String pathStr = file.toString();
	        Document doc = new Document();
	        
	        if (pathStr.contains("robots.txt")) {
	        	return;
	        }

	        doc.add(new KeywordField("path", pathStr, Field.Store.YES));
	        doc.add(new LongField("modified", lastModified, Field.Store.NO));

	        Parser parser = new Parser(new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8)));

	        String title = parser.title;
//	        System.out.println("title: " + title)¥
	        
	        doc.add(new TextField("title", title, Field.Store.YES));

	        String content = parser.body.replaceAll("\\s+", " ");;
//	        System.out.println("content: " + content);
	        doc.add(new TextField("contents", content, Field.Store.YES));

	        String url = pathStr.substring(docDir.toString().length() + 1).replace('\\', '/').replace("dummy", "");
//	        System.out.println("url: " + url);
	        doc.add(new StoredField("url", url));

	        if (demoEmbeddings != null) {
	            try (InputStream in = Files.newInputStream(file)) {
	                float[] vector =
	                        demoEmbeddings.computeEmbedding(
	                                new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8)));
	                doc.add(
	                        new KnnFloatVectorField(
	                                "contents-vector", vector, VectorSimilarityFunction.DOT_PRODUCT));
	            }
	        }

	        
	        String foundPrefecture = japanPrefectures.stream()
	            .filter(prefecture -> pathStr.contains(prefecture))
	            .findFirst()
	            .orElse(null);
	        
	        if (foundPrefecture == null) {
	            foundPrefecture = japanPrefectures.stream()
	                .filter(prefecture -> title.contains(prefecture))
	                .findFirst()
	                .orElse("-");
	            
	            if (foundPrefecture == null) {
	            	Set<String> mentionedPrefectures = japanPrefectures.stream()
	            			.filter(content::contains)
	            			.collect(Collectors.toSet());
	            	
	            	if (!mentionedPrefectures.isEmpty()) {
	            		foundPrefecture = mentionedPrefectures.size() > 12 ? "-" : String.join(", ", mentionedPrefectures);
	            	} else {
	            		foundPrefecture = "-";
	            	}
	            	
	            }
	        }

//	        System.out.println("prefecture: " + foundPrefecture);
	        doc.add(new StoredField("prefecture", foundPrefecture));

	        if (writer.getConfig().getOpenMode() == OpenMode.CREATE) {
	            System.out.println("adding " + file);
	            writer.addDocument(doc);
	        } else {
	            System.out.println("updating " + file);
	            writer.updateDocument(new Term("path", file.toString()), doc);
	        }
	    } catch (SAXException e) {
	        e.printStackTrace();
	    }
	}


  @Override
  public void close() throws IOException {
    IOUtils.close(vectorDict);
  }
}

//
//
///** Indexes a single document */
//void indexDoc(IndexWriter writer, Path file, long lastModified) throws IOException {
//  try (InputStream stream = Files.newInputStream(file)) {
//    // make a new, empty document
//    Document doc = new Document();
//
//    // Add the path of the file as a field named "path".  Use a
//    // field that is indexed (i.e. searchable), but don't tokenize
//    // the field into separate words and don't index term frequency
//    // or positional information:
//    doc.add(new KeywordField("path", file.toString(), Field.Store.YES));
//
//    // Add the last modified date of the file a field named "modified".
//    // Use a LongField that is indexed with points and doc values, and is efficient
//    // for both filtering (LongField#newRangeQuery) and sorting
//    // (LongField#newSortField).  This indexes to millisecond resolution, which
//    // is often too fine.  You could instead create a number based on
//    // year/month/day/hour/minutes/seconds, down the resolution you require.
//    // For example the long value 2011021714 would mean
//    // February 17, 2011, 2-3 PM.
//    doc.add(new LongField("modified", lastModified, Field.Store.NO));
//
//    // Add the contents of the file to a field named "contents".  Specify a Reader,
//    // so that the text of the file is tokenized and indexed, but not stored.
//    // Note that FileReader expects the file to be in UTF-8 encoding.
//    // If that's not the case searching for special characters will fail.
//    Parser parser = new Parser(new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8)));
//
//    String title = parser.title;
//    doc.add(new TextField("title", title, Field.Store.YES));
//
//    String content = parser.body.replaceAll("\\s+", " ");
//    doc.add(new TextField("contents", content, Field.Store.YES));
//
//    String url = "http://" + file.toString().substring(docDir.toString().length()+1).replace('\\', '/');
//    doc.add(new StoredField("url", url));
//    if (demoEmbeddings != null) {
//      try (InputStream in = Files.newInputStream(file)) {
//        float[] vector =
//            demoEmbeddings.computeEmbedding(
//                new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8)));
//        doc.add(
//            new KnnFloatVectorField(
//                "contents-vector", vector, VectorSimilarityFunction.DOT_PRODUCT));
//      }
//    }
//
//    if (writer.getConfig().getOpenMode() == OpenMode.CREATE) {
//      // New index, so we just add the document (no old document can be there):
//      System.out.println("adding " + file);
//      writer.addDocument(doc);
//    } else {
//      // Existing index (an old copy of this document may have been indexed) so
//      // we use updateDocument instead to replace the old one matching the exact
//      // path, if present:
//      System.out.println("updating " + file);
//      writer.updateDocument(new Term("path", file.toString()), doc);
//    }
//  } catch (SAXException e) {
//      // TODO Auto-generated catch block
//      e.printStackTrace();
//	}
//}