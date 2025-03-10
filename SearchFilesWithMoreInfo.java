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
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.th.ThaiAnalyzer;
import org.apache.lucene.demo.knn.DemoEmbeddings;
import org.apache.lucene.demo.knn.KnnVectorDict;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.highlight.Highlighter;
import org.apache.lucene.search.highlight.InvalidTokenOffsetsException;
import org.apache.lucene.search.highlight.QueryScorer;
import org.apache.lucene.search.highlight.SimpleFragmenter;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;

/** Simple command-line based search demo. */
public class SearchFilesWithMoreInfo {

//  private static List<String> japanPrefectures = List.of(
//		  "hokkaido", "aomori", "iwate", "miyagi", "akita", "yamagata", "fukushima",
//		  "ibaraki", "tochigi", "gunma", "saitama", "chiba", "tokyo", "kanagawa",
//		  "niigata", "toyama", "ishikawa", "fukui", "yamanashi", "nagano", "gifu",
//		  "shizuoka", "aichi", "mie", "shiga", "kyoto", "osaka", "hyogo", "nara",
//		  "wakayama", "tottori", "shimane", "okayama", "hiroshima", "yamaguchi",
//		  "tokushima", "kagawa", "ehime", "kochi", "fukuoka", "saga", "nagasaki",
//		  "kumamoto", "oita", "miyazaki", "kagoshima", "okinawa"
//		);
//	  

  private static List<String> japanPrefectures = Arrays.asList(
          "Hokkaido", "Aomori", "Iwate", "Miyagi", "Akita", "Yamagata", "Fukushima",
          "Ibaraki", "Tochigi", "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa",
          "Niigata", "Toyama", "Ishikawa", "Fukui", "Yamanashi", "Nagano", "Gifu",
          "Shizuoka", "Aichi", "Mie", "Shiga", "Kyoto", "Osaka", "Hyogo", "Nara",
          "Wakayama", "Tottori", "Shimane", "Okayama", "Hiroshima", "Yamaguchi",
          "Tokushima", "Kagawa", "Ehime", "Kochi", "Fukuoka", "Saga", "Nagasaki",
          "Kumamoto", "Oita", "Miyazaki", "Kagoshima", "Okinawa"
      );
	  
	
  private SearchFilesWithMoreInfo() {}
  private static Analyzer analyzer;

  /** Simple command-line based search demo. */
  public static void main(String[] args) throws Exception {
    String usage =
        "Usage:\tjava org.apache.lucene.demo.SearchFiles [-index dir] [-field f] [-repeat n] [-queries file] [-query string] [-raw] [-paging hitsPerPage] [-knn_vector knnHits]\n\nSee http://lucene.apache.org/core/9_0_0/demo/ for details.";
    if (args.length > 0 && ("-h".equals(args[0]) || "-help".equals(args[0]))) {
      System.out.println(usage);
      System.exit(0);
    }

    String index = "index";
    String field = "contents";
    String queries = null;
    int repeat = 0;
    boolean raw = false;
    int knnVectors = 0;
    String queryString = null;
    int hitsPerPage = 10;

    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "-index":
          index = args[++i];
          break;
        case "-field":
          field = args[++i];
          break;
        case "-queries":
          queries = args[++i];
          break;
        case "-query":
          queryString = args[++i];
          break;
        case "-repeat":
          repeat = Integer.parseInt(args[++i]);
          break;
        case "-raw":
          raw = true;
          break;
        case "-paging":
          hitsPerPage = Integer.parseInt(args[++i]);
          if (hitsPerPage <= 0) {
            System.err.println("There must be at least 1 hit per page.");
            System.exit(1);
          }
          break;
        case "-knn_vector":
          knnVectors = Integer.parseInt(args[++i]);
          break;
        default:
          System.err.println("Unknown argument: " + args[i]);
          System.exit(1);
      }
    }

    DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(index)));
    IndexSearcher searcher = new IndexSearcher(reader);
    analyzer = new SimpleAnalyzer();
    KnnVectorDict vectorDict = null;
    if (knnVectors > 0) {
      vectorDict = new KnnVectorDict(reader.directory(), IndexFiles.KNN_DICT);
    }
    BufferedReader in;
    if (queries != null) {
      in = Files.newBufferedReader(Paths.get(queries), StandardCharsets.UTF_8);
    } else {
      in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    }
    QueryParser parser = new QueryParser(field, analyzer);
    
    while (true) {
      if (queries == null && queryString == null) { // prompt the user
        System.out.println("Enter query: ");
      }

      String line = queryString != null ? queryString : in.readLine();

      if (line == null || line.length() == -1) {
        break;
      }

      line = line.trim();
      if (line.length() == 0) {
        break;
      }
      
      String searchField = "contents";

	  parser = new QueryParser(searchField, analyzer);
	  Query query = parser.parse(line);

      if (knnVectors > 0) {
        query = addSemanticQuery(query, vectorDict, knnVectors);
      }
      System.out.println("Searching for: " + query.toString(field));

      if (repeat > 0) { // repeat & time as benchmark
        Date start = new Date();
        for (int i = 0; i < repeat; i++) {
          searcher.search(query, 100);
        }
        Date end = new Date();
        System.out.println("Time: " + (end.getTime() - start.getTime()) + "ms");
      }

      doPagingSearch(in, searcher, query, hitsPerPage, raw, queries == null && queryString == null);

      if (queryString != null) {
        break;
      }
    }
    IOUtils.close(vectorDict, reader);
  }
  
  /**
   * This demonstrates a typical paging search scenario, where the search engine presents pages of
   * size n to the user. The user can then go to the next page if interested in the next hits.
   *
   * <p>When the query is executed for the first time, then only enough results are collected to
   * fill 5 result pages. If the user wants to page beyond this limit, then the query is executed
   * another time and all hits are collected.
   */
  public static void doPagingSearch(
      BufferedReader in,
      IndexSearcher searcher,
      Query query,
      int hitsPerPage,
      boolean raw,
      boolean interactive)
      throws IOException {

    QueryScorer queryScorer = new QueryScorer(query);
	Highlighter highlighter = new Highlighter(queryScorer);
	  
    // Collect enough docs to show 5 pages
    TopDocs results = searcher.search(query, 5 * hitsPerPage);
    ScoreDoc[] hits = results.scoreDocs;

    int numTotalHits = Math.toIntExact(results.totalHits.value());
    System.out.println(numTotalHits + " total matching documents");

    int start = 0;
    int end = Math.min(numTotalHits, hitsPerPage);

    while (true) {
      if (end > hits.length) {
        System.out.println(
            "Only results 1 - "
                + hits.length
                + " of "
                + numTotalHits
                + " total matching documents collected.");
        System.out.println("Collect more (y/n) ?");
        String line = in.readLine();
        if (line == null || line.length() == 0 || line.charAt(0) == 'n') {
          break;
        }

        hits = searcher.search(query, numTotalHits).scoreDocs;
      }

      end = Math.min(hits.length, start + hitsPerPage);

      StoredFields storedFields = searcher.storedFields();
      
      for (int i = start; i < end; i++) {
        if (raw) { // output raw format
          System.out.println("doc=" + hits[i].doc + " score=" + hits[i].score);
          continue;
        }

        Document doc = storedFields.document(hits[i].doc);
        String path = doc.get("path");
        if (path != null) {
		    String url = doc.get("url");
		    if (url != null) {
		    	System.out.println("\n" + (i + 1) + ". " + url);
		    }
			
		    String title = doc.get("title");
		    if (title != null) {
		    	System.out.println("   Title: " + title);
		    }
		
		    String content = doc.get("contents");
		    if (content != null) {
		    	TokenStream tokenStream = analyzer.tokenStream("contents", content);
		    	highlighter.setTextFragmenter(new SimpleFragmenter(100));
		    	try {
		    		String fragment = highlighter.getBestFragments(tokenStream, content, 2, "...");
		    		System.out.println("   Snippet: " + fragment);
		        } catch (InvalidTokenOffsetsException e) {
		          // TODO Auto-generated catch block
		        	e.printStackTrace();
		        }
		    }
          
		    String prefecture = doc.get("prefecture");
		    if (prefecture != null) {
		    	System.out.println("   Prefecture: " + prefecture);
		    } else {
		    	System.out.println("   Prefecture: -");
		    }
      
	    } else {
	    	System.out.println((i + 1) + ". " + "No path for this document");
	    }
	  }

      if (!interactive || end == 0) {
        break;
      }

      if (numTotalHits >= end) {
        boolean quit = false;
        while (true) {
          System.out.print("Press ");
          if (start - hitsPerPage >= 0) {
            System.out.print("(p)revious page, ");
          }
          if (start + hitsPerPage < numTotalHits) {
            System.out.print("(n)ext page, ");
          }
          System.out.println("(q)uit or enter number to jump to a page.");

          String line = in.readLine();
          if (line == null || line.length() == 0 || line.charAt(0) == 'q') {
            quit = true;
            break;
          }
          if (line.charAt(0) == 'p') {
            start = Math.max(0, start - hitsPerPage);
            break;
          } else if (line.charAt(0) == 'n') {
            if (start + hitsPerPage < numTotalHits) {
              start += hitsPerPage;
            }
            break;
          } else {
            int page = Integer.parseInt(line);
            if ((page - 1) * hitsPerPage < numTotalHits) {
              start = (page - 1) * hitsPerPage;
              break;
            } else {
              System.out.println("No such page");
            }
          }
        }
        if (quit) break;
        end = Math.min(numTotalHits, start + hitsPerPage);
      }
    }
  }

  private static Query addSemanticQuery(Query query, KnnVectorDict vectorDict, int k)
      throws IOException {
    StringBuilder semanticQueryText = new StringBuilder();
    QueryFieldTermExtractor termExtractor = new QueryFieldTermExtractor("contents");
    query.visit(termExtractor);
    for (String term : termExtractor.terms) {
      semanticQueryText.append(term).append(' ');
    }
    if (semanticQueryText.length() > 0) {
      KnnFloatVectorQuery knnQuery =
          new KnnFloatVectorQuery(
              "contents-vector",
              new DemoEmbeddings(vectorDict).computeEmbedding(semanticQueryText.toString()),
              k);
      BooleanQuery.Builder builder = new BooleanQuery.Builder();
      builder.add(query, BooleanClause.Occur.SHOULD);
      builder.add(knnQuery, BooleanClause.Occur.SHOULD);
      return builder.build();
    }
    return query;
  }

  private static class QueryFieldTermExtractor extends QueryVisitor {
    private final String field;
    private final List<String> terms = new ArrayList<>();

    QueryFieldTermExtractor(String field) {
      this.field = field;
    }

    @Override
    public boolean acceptField(String field) {
      return field.equals(this.field);
    }

    @Override
    public void consumeTerms(Query query, Term... terms) {
      for (Term term : terms) {
        this.terms.add(term.text());
      }
    }

    @Override
    public QueryVisitor getSubVisitor(BooleanClause.Occur occur, Query parent) {
      if (occur == BooleanClause.Occur.MUST_NOT) {
        return QueryVisitor.EMPTY_VISITOR;
      }
      return this;
    }
  }
}
